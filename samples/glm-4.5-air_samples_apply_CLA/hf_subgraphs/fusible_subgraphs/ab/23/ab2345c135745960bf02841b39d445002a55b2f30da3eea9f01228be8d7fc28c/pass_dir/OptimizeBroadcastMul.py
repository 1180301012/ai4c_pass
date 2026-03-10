import torch
import triton
import triton.language as tl

# Pattern matching function for broadcast multiplication
def pattern(in_2, tmp_4):
    tmp_5 = in_2 * tmp_4
    return tmp_5

# Argument extraction function
def replacement_args(in_2, tmp_4):
    return (in_2, tmp_4)

@triton.jit
def optimized_broadcast_mul_kernel(
    x_ptr,  # in_2 [1, 96, 128, 128]
    scale_ptr,  # tmp_4 [1, 96, 1, 1] 
    out_ptr,  # output [1, 96, 128, 128]
    batch_size, in_channels, in_height, in_width,
    scale_channels,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total_elements = batch_size * in_channels * in_height * in_width
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Convert flat offset to (batch, channels, height, width)
    residue = offsets
    b = residue // (in_channels * in_height * in_width)
    residue = residue % (in_channels * in_height * in_width)
    c = residue // (in_height * in_width)
    residue = residue % (in_height * in_width)
    h = residue // in_width
    w = residue % in_width
    
    # Load input value
    x_offset = b * in_channels * in_height * in_width + c * in_height * in_width + h * in_width + w
    x_val = tl.load(x_ptr + x_offset, mask=(b < batch_size) & (c < in_channels) & (h < in_height) & (w < in_width), other=0.0)
    
    # Load scale value (broadcast from [1, 96, 1, 1])
    # For this specific case, scale is the same for all spatial positions
    scale_offset = c  # since scale is [1, scale_channels, 1, 1] and we have same channels
    scale_val = tl.load(scale_ptr + scale_offset, mask=c < scale_channels, other=1.0)
    
    # Multiplication
    out_val = x_val * scale_val
    
    # Store result
    tl.store(out_ptr + offsets, out_val, mask=mask)

@torch.fx.wrap
def optimized_broadcast_mul(in_2, tmp_4):
    # Get input shapes
    batch_size, in_channels, in_height, in_width = in_2.shape
    scale_channels = tmp_4.shape[1]  # Should be same as in_channels
    
    # Prepare output tensor
    out = torch.empty((batch_size, in_channels, in_height, in_width), dtype=in_2.dtype, device=in_2.device)
    
    # Tile size - optimized for large spatial dimensions
    BLOCK_SIZE = 1024
    
    # Calculate number of programs needed
    total_elements = batch_size * in_channels * in_height * in_width
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    optimized_broadcast_mul_kernel[(num_programs,)](
        x_ptr=in_2,
        scale_ptr=tmp_4,
        out_ptr=out,
        batch_size=batch_size,
        in_channels=in_channels,
        in_height=in_height,
        in_width=in_width,
        scale_channels=scale_channels,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function
def replacement_func():
    return optimized_broadcast_mul