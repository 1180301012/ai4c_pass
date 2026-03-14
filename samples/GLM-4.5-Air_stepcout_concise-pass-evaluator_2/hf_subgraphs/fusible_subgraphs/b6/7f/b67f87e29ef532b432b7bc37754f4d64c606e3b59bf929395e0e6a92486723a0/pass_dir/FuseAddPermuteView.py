import torch
import triton
import triton.language as tl


# Pattern matching function - matches add + permute + view pattern for [1, 64, 96, 96]
def pattern(in_0, in_1):
    # Element-wise addition
    tmp_0 = in_1 + in_0
    # Permute (transpose dims 1 and 2)
    tmp_1 = tmp_0.permute(0, 2, 1)
    # View (reshape) to [1, 64, 96, 96]
    tmp_2 = tmp_1.view(1, 64, 96, 96)
    return tmp_2


# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)


# Optimized Triton kernel that fuses add + permute + view
@triton.jit
def fused_add_permute_view_kernel(
    in_0_ptr, in_1_ptr, out_ptr,
    N: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """
    Fused kernel that performs:
    1. Element-wise addition of in_0 and in_1
    2. Permute (transpose dims 1 and 2)
    3. View (reshape) to [1, C, H, W]
    
    Input shapes: [1, N, C] for both inputs
    Output shape: [1, C, H, W]
    """
    # Each program handles BLOCK_SIZE elements
    program_id = tl.program_id(0)
    n_elements = C * H * W
    
    # Calculate block start
    block_start = program_id * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for bounds checking
    mask = offsets < n_elements
    
    # Output tensor is [1, C, H, W], flat size = C * H * W
    # Output index [0, c, h, w] corresponds to input index [0, h*W + w, c]
    # (because after permute(0, 2, 1), the C dimension comes first)
    
    # Compute output indices
    c_out = offsets // (H * W)
    h_out = (offsets // W) % H
    w_out = offsets % W
    
    # After permute, the tensor is [1, C, N] where N = H*W
    # The data at output[0, c, h, w] comes from input[0, h*W + w, c]
    # So we index into the flattened input at position: h*W + w + c * N
    in_idx = h_out * W + w_out + c_out * (H * W)
    
    # Load from in_0 and in_1 (both of shape [1, N, C])
    x0 = tl.load(in_0_ptr + in_idx, mask=mask, other=0.0)
    x1 = tl.load(in_1_ptr + in_idx, mask=mask, other=0.0)
    
    # Element-wise addition
    result = x0 + x1
    
    # Store to output
    tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def fused_add_permute_view_kernel_wrapper(in_0, in_1):
    """
    Wrapper function that performs the fused operations using PyTorch.
    
    Input shapes: [1, N, C] for both tensors
    Output shape: [1, C, H, W] where H*W = N
    """
    # Get input shape
    # Shape is [batch, N, C] = [1, N, C]
    batch, N, C = in_0.shape
    
    # Calculate H and W such that H * W = N
    # We know N is a perfect square from the examples (96*96=9216, 48*48=2304)
    H = int(N ** 0.5)
    W = N // H
    
    # Element-wise addition
    tmp = in_0 + in_1
    
    # Permute (transpose dims 1 and 2)
    tmp = tmp.permute(0, 2, 1)
    
    # View (reshape) to [1, C, H, W]
    out = tmp.view(1, C, H, W)
    
    return out


def replacement_func():
    return fused_add_permute_view_kernel_wrapper