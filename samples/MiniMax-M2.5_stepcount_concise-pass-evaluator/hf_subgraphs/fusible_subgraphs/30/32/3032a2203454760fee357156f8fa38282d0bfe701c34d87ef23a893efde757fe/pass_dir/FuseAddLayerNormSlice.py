import torch
import triton
import triton.language as tl

# The model has 4 inputs: in_0 (bias), in_1 (weight), in_2, in_3
# Pattern matches layer_norm + slice, where in_2 is the result of (in_2 + in_3)
# This works because the subgraph matcher finds the layer_norm node which takes tmp_2 as input
# and tmp_2 = in_2 + in_3 in the original graph

def pattern(in_0, in_1, in_2):
    """
    Match the pattern: layer_norm + slice.
    
    in_0: bias [512]
    in_1: weight [512]  
    in_2: input tensor [1, 145, 512] - the result of in_2 + in_3 from the original model
    
    Returns: tmp_8, shape [1, 512]
    """
    tmp_7 = torch.nn.functional.layer_norm(in_2, (512,), in_1, in_0, 1e-06)
    tmp_8 = tmp_7[slice(None, None, None), 0]
    return tmp_8


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# Autotune configuration for better performance
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_stages=5, num_warps=2),
    ],
    key=['H'],
)
@triton.jit
def fused_add_layernorm_slice_kernel(
    in_ptr, in_ptr_dummy, in_1_ptr, in_0_ptr,
    out_ptr,
    N, C, H,
    stride_in_0, stride_in_1, stride_in_2,
    stride_in2_0, stride_in2_1, stride_in2_2,
    stride_out_0, stride_out_1,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized kernel for layer_norm + slice.
    
    Input shape: [N, C, H] = [1, 145, 512]
    layer_norm normalizes over H=512 for each (n, c) position.
    Output: tmp_7[:, 0, :] = shape [1, 512]
    
    We process each (n, c) pair that will contribute to the output.
    Since output is tmp_7[:, 0, :], we only need c=0.
    So we process c=0, and for each h in [0, 512).
    """
    # Get position
    # We need to process c=0 for all h in [0, H)
    # But we also need the full statistics across H for each c
    pid = tl.program_id(0)
    h_idx = pid % H  # Each program handles one h position
    c_idx = 0  # We only need c=0 for the output
    
    if h_idx >= H:
        return
    
    # For layer_norm, we need to compute mean and variance across H=512 for each (n, c)
    # Since we're computing output for c=0, we need mean and var across H for c=0
    
    # First pass: compute mean across H for this c
    sum_val = 0.0
    for h in range(H):
        in_offset = c_idx * stride_in_1 + h * stride_in_2
        val = tl.load(in_ptr + in_offset)
        sum_val += val
    
    mean = sum_val / tl.cast(H, tl.float32)
    
    # Second pass: compute variance across H
    sum_sq = 0.0
    for h in range(H):
        in_offset = c_idx * stride_in_1 + h * stride_in_2
        val = tl.load(in_ptr + in_offset)
        diff = val - mean
        sum_sq += diff * diff
    
    variance = sum_sq / tl.cast(H, tl.float32)
    std = tl.sqrt(variance + eps)
    
    # Get the value at (c=0, h=h_idx) and normalize
    in_offset = c_idx * stride_in_1 + h_idx * stride_in_2
    val = tl.load(in_ptr + in_offset)
    
    # Normalize: (x - mean) / std
    normalized = (val - mean) / std
    
    # Load weight and bias for this h
    weight = tl.load(in_1_ptr + h_idx)
    bias = tl.load(in_0_ptr + h_idx)
    
    # Apply weight and bias
    result = normalized * weight + bias
    
    # Store to output
    out_offset = h_idx
    tl.store(out_ptr + out_offset, result)


@torch.fx.wrap
def fused_add_layernorm_slice_wrapper(in_0, in_1, in_2):
    """
    Wrapper function that launches the Triton kernel.
    
    Note: in_2 is already the result of (in_2 + in_3) from the original model.
    The add operation has already been applied by the time we get here.
    
    Inputs:
    - in_0: bias, shape [512]
    - in_1: weight, shape [512]
    - in_2: shape [1, 145, 512] - already includes the add result
    
    Output:
    - shape [1, 512]
    """
    N, C, H = in_2.shape  # [1, 145, 512]
    eps = 1e-06
    
    # Move weights to GPU if needed
    if not in_0.is_cuda:
        in_0 = in_0.cuda()
    if not in_1.is_cuda:
        in_1 = in_1.cuda()
    if not in_2.is_cuda:
        in_2 = in_2.cuda()
    
    # Allocate output
    output = torch.empty((N, H), dtype=torch.float32, device='cuda')
    
    # Launch kernel
    # Grid: one program per h position (512 programs)
    grid = (H,)
    
    # For this simplified version, we don't redo the add
    # We just do layer_norm + slice
    fused_add_layernorm_slice_kernel[grid](
        in_2, in_2, in_1, in_0,  # Pass in_2 twice since we don't have in_3
        output,
        N, C, H,
        in_2.stride(0), in_2.stride(1), in_2.stride(2),
        in_2.stride(0), in_2.stride(1), in_2.stride(2),  # Same strides for both
        output.stride(0), output.stride(1),
        eps,
    )
    
    return output


def replacement_func():
    """Return the replacement function."""
    return fused_add_layernorm_slice_wrapper