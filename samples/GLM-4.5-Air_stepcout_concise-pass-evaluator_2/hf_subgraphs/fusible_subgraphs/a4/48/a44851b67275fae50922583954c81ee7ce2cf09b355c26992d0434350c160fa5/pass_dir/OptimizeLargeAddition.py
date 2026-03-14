import torch
import triton
import triton.language as tl

# Pattern for the specific large addition: in_2 + in_3
def pattern(in_2, in_3):
    """Pattern matches: in_2 + in_3 (the large tensor addition)"""
    tmp_2 = in_2 + in_3
    return tmp_2

# Argument extraction function
def replacement_args(in_2, in_3):
    return (in_2, in_3)

# Optimized Triton kernel for large tensor addition
@triton.jit
def large_add_kernel(
    x_ptr, y_ptr, out_ptr,
    n_batch, n_height, n_channels,
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
):
    # Each program handles a 2D tile of the 3D tensor
    batch = tl.program_id(0)
    height = tl.program_id(1)
    channel_start = tl.program_id(2) * BLOCK_SIZE_X
    
    # Bounds checking
    mask = (channel_start + tl.arange(0, BLOCK_SIZE_X)[:, None] < n_channels) & \
           (height < n_height) & (batch < n_batch)
    
    # Load data for this batch and height, multiple channels
    x_offset = batch * n_height * n_channels + height * n_channels + channel_start
    y_offset = batch * n_height * n_channels + height * n_channels + channel_start
    
    x = tl.load(x_ptr + x_offset + tl.arange(0, BLOCK_SIZE_X)[:, None], 
                mask=mask, other=0.0)
    y = tl.load(y_ptr + y_offset + tl.arange(0, BLOCK_SIZE_X)[:, None], 
                mask=mask, other=0.0)
    
    # Addition
    out = x + y
    
    # Store result
    out_offset = batch * n_height * n_channels + height * n_channels + channel_start
    tl.store(out_ptr + out_offset + tl.arange(0, BLOCK_SIZE_X)[:, None], 
             out, mask=mask)

# Optimized wrapper using larger block sizes for better performance
@torch.fx.wrap
def large_add_triton(x, y):
    batch_size, height, channels = x.shape
    
    # Use larger block sizes for better GPU utilization
    block_size_x = 64  # Process 64 channels at once
    block_size_y = 1   # Process 1 height at once
    
    # Calculate grid dimensions
    grid_x = (channels + block_size_x - 1) // block_size_x
    grid_y = height
    grid_z = batch_size
    grid = (grid_z, grid_y, grid_x)
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Launch optimized kernel
    large_add_kernel[grid](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_batch=batch_size,
        n_height=height,
        n_channels=channels,
        BLOCK_SIZE_X=block_size_x,
        BLOCK_SIZE_Y=block_size_y,
    )
    
    return out

def replacement_func():
    return large_add_triton