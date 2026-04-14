import torch
import triton
import triton.language as tl

# Pattern matching function for mean operation
def pattern(x):
    """Match a simple mean operation"""
    return x.mean((2, 3))

# Argument extraction function
def replacement_args(x):
    return (x,)

# Triton kernel for computing mean over spatial dimensions
@triton.jit
def mean_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    num_channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    if pid >= batch_size * num_channels:
        return
    
    # Calculate indices
    batch_idx = pid // num_channels
    channel_idx = pid % num_channels
    
    # Calculate mean for this channel
    spatial_elements = height * width
    channel_sum = 0.0
    
    # Sum all spatial elements
    for h in range(height):
        for w in range(width):
            ptr = input_ptr + (batch_idx * num_channels * height * width + 
                              channel_idx * height * width + h * width + w)
            val = tl.load(ptr)
            channel_sum += val
    
    # Compute mean
    channel_mean = channel_sum / spatial_elements
    
    # Store result
    output_ptr_offset = batch_idx * num_channels + channel_idx
    tl.store(output_ptr + output_ptr_offset, channel_mean)

@torch.fx.wrap
def triton_mean(x):
    """Mean operation using Triton kernel"""
    batch_size, num_channels, height, width = x.shape
    
    # Output shape: [batch_size, num_channels]
    output = torch.empty(batch_size, num_channels, dtype=x.dtype, device=x.device)
    
    BLOCK_SIZE = 1024
    grid = ((batch_size * num_channels + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    mean_kernel[grid](
        input_ptr=x,
        output_ptr=output,
        batch_size=batch_size,
        num_channels=num_channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Replacement function (must return a callable function)
def replacement_func():
    return triton_mean