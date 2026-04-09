import torch
import triton
import triton.language as tl

@triton.jit
def optimized_max_pool2d_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    channels,
    height_in,
    width_in,
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
):
    # Get program IDs
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    
    # Calculate spatial blocks
    y_blocks = (height_in + BLOCK_SIZE_Y - 1) // BLOCK_SIZE_Y
    x_blocks = (width_in + BLOCK_SIZE_Y - 1) // BLOCK_SIZE_Y
    block_y = tl.program_id(2) // x_blocks
    block_x = tl.program_id(2) % x_blocks
    
    # Each program handles a BLOCK_SIZE_Y x BLOCK_SIZE_Y region
    y_idx = block_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    x_idx = block_x * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    
    # Create masks for input indices
    y_mask = y_idx < height_in
    x_mask = x_idx < width_in
    
    # Load the 2x2 window for each thread
    window_vals = tl.zeros((BLOCK_SIZE_Y, BLOCK_SIZE_Y), dtype=tl.float32)
    for i in range(BLOCK_SIZE_Y):
        for j in range(BLOCK_SIZE_Y):
            if y_mask[i] and x_mask[j]:
                # Calculate input coordinates (with stride=1, kernel_size=2, padding=0, dilation=1)
                y_in = min(block_y * BLOCK_SIZE_Y + i, height_in - 1)
                x_in = min(block_x * BLOCK_SIZE_Y + j, width_in - 1)
                
                # Load input value
                x_val = tl.load(
                    x_ptr + batch_idx * channels * height_in * width_in + 
                    channel_idx * height_in * width_in + 
                    y_in * width_in + x_in,
                    other=-float('inf')
                )
                window_vals[i, j] = x_val
    
    # Find max in 2x2 window (stride=1, so we need to check overlapping windows)
    max_vals = tl.zeros(BLOCK_SIZE_Y * BLOCK_SIZE_Y, dtype=tl.float32)
    idx = 0
    for i in range(BLOCK_SIZE_Y - 1):
        for j in range(BLOCK_SIZE_Y - 1):
            if y_idx[i:i+2].max() + 1 < height_in and x_idx[j:j+2].max() + 1 < width_in:
                # Get 2x2 window
                window = window_vals[i:i+2, j:j+2]
                max_val = tl.max(window)
                
                # Determine output position (using ceil_mode)
                y_out = (block_y * BLOCK_SIZE_Y + i + 1 + 1) // 2 - 1  # ceil division
                x_out = (block_x * BLOCK_SIZE_Y + j + 1 + 1) // 2 - 1  # ceil division
                
                if y_out < (height_in + 1) // 2 and x_out < (width_in + 1) // 2:
                    idx2d = y_out * ((width_in + 1) // 2) + x_out
                    max_vals[idx] = max_val
            idx += 1
    
    # Store output
    out_width = (width_in + 1) // 2
    for i in range(min(BLOCK_SIZE_Y * BLOCK_SIZE_Y, len(max_vals))):
        y_out = i // out_width
        x_out = i % out_width
        if y_out < ((height_in + 1) // 2) and x_out < ((width_in + 1) // 2):
            tl.store(
                out_ptr + batch_idx * channels * ((height_in + 1) // 2) * ((width_in + 1) // 2) + 
                channel_idx * ((height_in + 1) // 2) * ((width_in + 1) // 2) + 
                y_out * ((width_in + 1) // 2) + x_out,
                max_vals[i]
            )

@torch.fx.wrap
def optimized_max_pool2d(x):
    batch_size, channels, height, width = x.shape
    
    # Optimal block sizes for spatial dimensions
    BLOCK_SIZE_X = 16
    BLOCK_SIZE_Y = 16
    
    # Calculate output dimensions with ceil_mode=True
    out_height = (height + 1) // 2  # ceil division
    out_width = (width + 1) // 2   # ceil division
    
    output = torch.empty(batch_size, channels, out_height, out_width, dtype=x.dtype, device=x.device)
    
    # Calculate grid dimensions
    num_blocks_y = (height + BLOCK_SIZE_Y - 1) // BLOCK_SIZE_Y
    num_blocks_x = (width + BLOCK_SIZE_Y - 1) // BLOCK_SIZE_Y
    spatial_blocks = num_blocks_y * num_blocks_x
    
    optimized_max_pool2d_kernel[
        (batch_size, channels, spatial_blocks)
    ](
        x_ptr=x,
        out_ptr=output,
        batch_size=batch_size,
        channels=channels,
        height_in=height,
        width_in=width,
        BLOCK_SIZE_X=BLOCK_SIZE_X,
        BLOCK_SIZE_Y=BLOCK_SIZE_Y
    )
    
    return output

def pattern(x):
    """Optimized max_pool2d with kernel_size=2, stride=1, padding=0, dilation=1, ceil_mode=True"""
    return torch.nn.functional.max_pool2d(x, 2, 1, 0, 1, ceil_mode=True, return_indices=False)

def replacement_args(x):
    return (x,)

def replacement_func():
    return optimized_max_pool2d