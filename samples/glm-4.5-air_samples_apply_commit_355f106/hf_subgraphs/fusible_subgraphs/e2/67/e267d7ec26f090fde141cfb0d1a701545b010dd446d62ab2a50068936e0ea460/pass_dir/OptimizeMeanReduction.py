import torch
import triton
import triton.language as tl

# Pattern matching function: matches mean reduction over spatial dimensions
def pattern(x):
    return x.mean((2, 3), keepdim=True)

# Argument extraction function
def replacement_args(x):
    return (x,)

# Optimized Triton kernel for mean reduction over spatial dimensions
@triton.jit
def mean_reduction_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE_Y: tl.constexpr,
    BLOCK_SIZE_X: tl.constexpr,
):
    # Each thread block processes a spatial tile
    pid_y = tl.program_id(0)
    pid_x = tl.program_id(1)
    pid_b = tl.program_id(2)
    pid_c = tl.program_id(3)
    
    # Grid boundaries
    y_start = pid_y * BLOCK_SIZE_Y
    x_start = pid_x * BLOCK_SIZE_X
    y_end = min(y_start + BLOCK_SIZE_Y, height)
    x_end = min(x_start + BLOCK_SIZE_X, width)
    
    # Initialize accumulator for this block
    sum_val = 0.0
    
    # Compute the number of elements in this block
    block_elements = (y_end - y_start) * (x_end - x_start)
    
    # Loop over spatial dimensions with vectorization
    y_offset = y_start + tl.arange(0, BLOCK_SIZE_Y)
    x_offset = x_start + tl.arange(0, BLOCK_SIZE_X)
    
    # Create 2D coordinates
    ys, xs = tl.meshgrid(y_offset, x_offset)
    
    # Flatten spatial coordinates for contiguous access
    flat_offsets = ys * width + xs
    
    # Process each channel and batch element
    if pid_b < batch_size and pid_c < channels:
        # Loop through the spatial block
        for y in range(y_start, y_end):
            for x in range(x_start, x_end):
                # Calculate global offset for current spatial position
                spatial_offset = y * width + x
                batch_channel_offset = pid_b * (channels * height * width) + pid_c * (height * width) + spatial_offset
                
                # Load and accumulate
                val = tl.load(x_ptr + batch_channel_offset, other=0.0)
                sum_val += val
        
        # Compute mean for this block
        mean_val = sum_val / (block_elements + 1e-6)  # Add small epsilon to prevent division by zero
        
        # Store result - output has shape [batch_size, channels, 1, 1]
        output_offset = pid_b * (channels * 1 * 1) + pid_c * (1 * 1)
        tl.store(out_ptr + output_offset, mean_val)

# Kernel wrapper with autotune
@torch.fx.wrap
def optimized_mean_reduction(x):
    batch_size, channels, height, width = x.shape
    
    # Choose block sizes based on input dimensions
    if height <= 32 and width <= 32:
        BLOCK_SIZE_Y = 16
        BLOCK_SIZE_X = 16
    elif height <= 64 and width <= 64:
        BLOCK_SIZE_Y = 32  
        BLOCK_SIZE_X = 32
    else:
        BLOCK_SIZE_Y = 64
        BLOCK_SIZE_X = 64
    
    # Calculate grid dimensions
    grid_y = (height + BLOCK_SIZE_Y - 1) // BLOCK_SIZE_Y
    grid_x = (width + BLOCK_SIZE_X - 1) // BLOCK_SIZE_X
    grid_b = batch_size
    grid_c = channels
    
    # Create output tensor with keepdim=True
    out_shape = (batch_size, channels, 1, 1)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    # Launch kernel
    mean_reduction_kernel[
        (grid_y, grid_x, grid_b, grid_c)
    ](
        x,
        out,
        batch_size,
        channels,
        height,
        width,
        BLOCK_SIZE_Y=BLOCK_SIZE_Y,
        BLOCK_SIZE_X=BLOCK_SIZE_X,
    )
    
    return out

# Replacement function
def replacement_func():
    return optimized_mean_reduction