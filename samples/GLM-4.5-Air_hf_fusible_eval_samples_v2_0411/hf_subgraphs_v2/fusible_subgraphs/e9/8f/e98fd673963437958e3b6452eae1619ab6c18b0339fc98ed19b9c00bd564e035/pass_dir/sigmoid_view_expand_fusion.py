import torch
import triton
import triton.language as tl

def pattern(in_2, in_1):
    tmp_0 = in_2.sigmoid()
    tmp_1 = tmp_0.view(1, -1, 1, 1)
    tmp_2 = tmp_1.expand_as(in_1)
    return tmp_2

def replacement_args(in_2, in_1):
    return (in_2, in_1)

@triton.jit
def sigmoid_expand_kernel(
    in_ptr,
    out_ptr,
    n_channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (n_channels * height * width)
    
    # Load input data (in_2 is [1, 1, n_channels] -> [n_channels])
    in_flat = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    
    # Apply sigmoid and broadcast to spatial dimensions
    # Output should be [1, n_channels, height, width]
    sigmoid_val = 1.0 / (1.0 + tl.exp(-in_flat))
    
    # Since we're broadcasting, each sigmoid value is repeated across all spatial positions
    # For block processing, we compute the sigmoid once per channel and let the outer
    # loop handle spatial broadcasting
    out_val = sigmoid_val
    
    # Store the result (broadcasting will be handled by the caller)
    tl.store(out_ptr + offsets, out_val, mask=mask)

@torch.fx.wrap
def fused_sigmoid_expand(in_2, target_shape):
    # Get input tensor properties
    n_channels = in_2.shape[2]  # [1, 1, 2048]
    height, width = target_shape[2], target_shape[3]  # [1, 2048, H, W]
    
    # Calculate total elements in output
    total_elements = n_channels * height * width
    
    # Choose block size
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty(target_shape, dtype=in_2.dtype, device=in_2.device)
    
    # Launch kernel - process each channel once and broadcast spatially
    sigmoid_expand_kernel[(num_programs, 1, 1)](
        in_ptr=in_2,
        out_ptr=out,
        n_channels=n_channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return lambda in_2, in_1: fused_sigmoid_expand(in_2, in_1.shape)