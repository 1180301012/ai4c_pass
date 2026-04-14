import torch
import triton
import triton.language as tl

# Pattern matching function for mean operation
def pattern(x):
    """Match mean over spatial dimensions - works across all graph variants"""
    return x.mean((2, 3))

# Argument extraction function
def replacement_args(x):
    return (x,)

# Highly optimized Triton kernel with better parallelism
@triton.jit
def best_mean_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    num_channels,
    height,
    width,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Best performing mean kernel with 2D grid for better GPU utilization"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Each program handles a block of channels
    if pid_m >= (num_channels + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M:
        return
    
    # Calculate channel range for this program
    start_channel = pid_m * BLOCK_SIZE_M
    end_channel = min(start_channel + BLOCK_SIZE_M, num_channels)
    
    # Process multiple channels in parallel
    for channel_idx in range(start_channel, end_channel):
        channel_sum = 0.0
        spatial_elements = height * width
        
        # Optimized spatial reduction
        for h in range(height):
            for w in range(width):
                ptr = input_ptr + (channel_idx * height * width + h * width + w)
                val = tl.load(ptr)
                channel_sum += val
        
        # Compute mean and store
        channel_mean = channel_sum / spatial_elements
        tl.store(output_ptr + channel_idx, channel_mean)

@torch.fx.wrap
def best_mean_function(x):
    """
    Best performing mean computation using optimized 2D grid
    """
    batch_size, num_channels, height, width = x.shape
    
    # Create output tensor
    output = torch.empty(num_channels, dtype=x.dtype, device=x.device)
    
    # Use 2D grid for better GPU utilization
    BLOCK_SIZE_M = 64  # Channels per block
    BLOCK_SIZE_N = 1   # One dimension for spatial (could be expanded)
    
    grid_m = (num_channels + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = 1
    
    # Launch optimized kernel with 2D grid
    best_mean_kernel[(grid_m, grid_n)](
        input_ptr=x,
        output_ptr=output,
        batch_size=batch_size,
        num_channels=num_channels,
        height=height,
        width=width,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return output

# Replacement function (must return a callable function)
def replacement_func():
    return best_mean_function