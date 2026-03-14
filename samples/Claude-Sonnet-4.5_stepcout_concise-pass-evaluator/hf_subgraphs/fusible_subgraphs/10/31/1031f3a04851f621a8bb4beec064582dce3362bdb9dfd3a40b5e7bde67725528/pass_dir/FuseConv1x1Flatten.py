import torch
import triton
import triton.language as tl

def pattern(bias, weight, input):
    """
    Match conv2d (1x1 kernel) followed by flatten.
    Must match the exact call signature from model.py with positional arguments.
    """
    conv_out = torch.conv2d(input, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    flat_out = torch.flatten(conv_out, 2)
    return flat_out

def replacement_args(bias, weight, input):
    return (bias, weight, input)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64, 'BLOCK_C': 16}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64, 'BLOCK_C': 32}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128, 'BLOCK_C': 16}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128, 'BLOCK_C': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128, 'BLOCK_C': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256, 'BLOCK_C': 16}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256, 'BLOCK_C': 32}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256, 'BLOCK_C': 64}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512, 'BLOCK_C': 16}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512, 'BLOCK_C': 32}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512, 'BLOCK_C': 64}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024, 'BLOCK_C': 16}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024, 'BLOCK_C': 32}, num_warps=8),
    ],
    key=['N', 'C_in'],
)
@triton.jit
def fused_conv1x1_flatten_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    B, C_in, C_out, HW, N,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """
    Optimized fused kernel for 1x1 convolution + flatten.
    Uses vectorized operations over channel blocks to avoid Python loops.
    """
    pid = tl.program_id(0)
    
    # Compute which output elements this block handles
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Decode the flattened index: N = B * C_out * HW
    # offsets = b * (C_out * HW) + c_out * HW + hw
    b_idx = offsets // (C_out * HW)
    remainder = offsets % (C_out * HW)
    c_out_idx = remainder // HW
    hw_idx = remainder % HW
    
    # Initialize accumulator
    result = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Loop over input channels with blocking
    for c_in_start in range(0, C_in, BLOCK_C):
        # Channel offsets for this block
        c_in_offsets = c_in_start + tl.arange(0, BLOCK_C)
        c_in_mask = c_in_offsets < C_in
        
        # Load input for all channels in this block: [BLOCK_SIZE, BLOCK_C]
        # For each output position, load BLOCK_C input channels
        # input[b_idx, c_in_offsets, hw_idx] but expanded
        input_base = b_idx * (C_in * HW) + hw_idx
        input_offsets_2d = input_base[:, None] + c_in_offsets[None, :] * HW
        input_block = tl.load(input_ptr + input_offsets_2d, 
                             mask=mask[:, None] & c_in_mask[None, :], 
                             other=0.0)
        
        # Load weight for all channels in this block: [BLOCK_SIZE, BLOCK_C]
        # For each output position, load BLOCK_C weight channels
        # weight[c_out_idx, c_in_offsets]
        weight_base = c_out_idx * C_in
        weight_offsets_2d = weight_base[:, None] + c_in_offsets[None, :]
        weight_block = tl.load(weight_ptr + weight_offsets_2d,
                              mask=mask[:, None] & c_in_mask[None, :],
                              other=0.0)
        
        # Multiply and sum over the channel dimension
        result += tl.sum(input_block * weight_block, axis=1)
    
    # Add bias: bias[c_out_idx]
    bias_vals = tl.load(bias_ptr + c_out_idx, mask=mask, other=0.0)
    result += bias_vals
    
    # Store output: output[b_idx, c_out_idx, hw_idx]
    output_offset = b_idx * (C_out * HW) + c_out_idx * HW + hw_idx
    tl.store(output_ptr + output_offset, result, mask=mask)

@torch.fx.wrap
def fused_conv1x1_flatten(bias, weight, input):
    """
    Wrapper function for the fused conv1x1 + flatten kernel.
    """
    B, C_in, H, W = input.shape
    C_out = weight.shape[0]
    HW = H * W
    N = B * C_out * HW  # Total number of output elements
    
    # Allocate output tensor in flattened format
    output = torch.empty((B, C_out, HW), device=input.device, dtype=input.dtype)
    
    # Choose a default block size (autotuner will optimize this)
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(N, BLOCK_SIZE)
    
    # Launch kernel with 1D grid
    grid = (num_blocks,)
    
    fused_conv1x1_flatten_kernel[grid](
        input,
        weight,
        bias,
        output,
        B, C_in, C_out, HW, N,
    )
    
    return output

def replacement_func():
    return fused_conv1x1_flatten