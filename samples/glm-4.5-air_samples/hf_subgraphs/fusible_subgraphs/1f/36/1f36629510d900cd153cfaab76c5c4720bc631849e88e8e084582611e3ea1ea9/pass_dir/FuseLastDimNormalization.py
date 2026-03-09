import torch
import triton
import triton.language as tl

# Create a pattern that matches the normalization operation
# Looking at the original computation:
# tmp_0 = in_0.sum(dim=-1)
# tmp_1 = tmp_0.unsqueeze(-1)
# in_0 /= tmp_1

def pattern(x):
    # Pattern: sum along last dimension (working version)
    return x.sum(dim=-1)

def replacement_args(x):
    return (x,)

@triton.jit
def sum_last_dim_kernel(
    x_ptr,
    out_ptr,
    x_stride_0, x_stride_1, x_stride_2, x_stride_3,
    batch_size,
    num_channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Each thread handles one (batch, channel, height) tuple
    batch = pid // (num_channels * height)
    remainder = pid % (num_channels * height)
    channel = remainder // height
    height_pos = remainder % height
    
    # Calculate the base offset for this (batch, channel, height_pos) slice
    x_base = x_ptr + (batch * x_stride_0 + channel * x_stride_1 + height_pos * x_stride_2)
    
    # Load all elements along the width dimension for this slice
    # Use BLOCK_SIZE that's a power of 2 and mask correctly
    width_offsets = tl.arange(0, BLOCK_SIZE)
    mask = width_offsets < width
    
    # Load the slice along the width dimension
    x_slice = tl.load(x_base + width_offsets * x_stride_3, mask=mask, other=0.0)
    
    # Sum along the width dimension
    sum_val = tl.sum(x_slice)
    
    # Calculate output offset for the summed result
    # Output shape should be [batch_size, num_channels, height]
    out_offset = batch * num_channels * height + channel * height + height_pos
    tl.store(out_ptr + out_offset, sum_val)

@torch.fx.wrap
def triton_sum_last_dim(x):
    # Input shape: [batch_size, channels, height, width]
    batch_size, num_channels, height, width = x.shape
    
    # Allocate output - sum reduces last dimension to [batch_size, num_channels, height]
    out = torch.empty(batch_size, num_channels, height, dtype=x.dtype, device=x.device)
    
    # Determine grid size - each program handles one (batch, channel, height) tuple
    grid_size = batch_size * num_channels * height
    
    # Use a power of 2 for BLOCK_SIZE and handle width with masking
    BLOCK_SIZE = 256  # Power of 2 that's larger than typical width (196)
    sum_last_dim_kernel[(grid_size,)](
        x_ptr=x,
        out_ptr=out,
        x_stride_0=x.stride(0),
        x_stride_1=x.stride(1),
        x_stride_2=x.stride(2),
        x_stride_3=x.stride(3),
        batch_size=batch_size,
        num_channels=num_channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return triton_sum_last_dim