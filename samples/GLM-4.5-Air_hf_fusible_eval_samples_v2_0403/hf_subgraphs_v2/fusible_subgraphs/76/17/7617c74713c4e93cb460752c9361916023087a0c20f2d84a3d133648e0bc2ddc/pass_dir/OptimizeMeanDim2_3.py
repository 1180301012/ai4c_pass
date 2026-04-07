import torch
import triton
import triton.language as tl

# Pattern matching function - matches mean computation across spatial dimensions
def pattern(input_tensor):
    mean_out = input_tensor.mean((2, 3), keepdim=False)
    return mean_out

# Argument extraction function
def replacement_args(input_tensor):
    return (input_tensor,)

# Optimized mean kernel for spatial dimensions (2, 3)
@triton.jit
def optimized_mean_kernel(
    input_ptr,
    output_ptr,
    n_channels,  # Number of channels (dimension 1)
    height,      # Original height (dimension 2)
    width,       # Original width (dimension 3)
    n_batches,   # Number of batches (dimension 0)
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one channel
    pid = tl.program_id(0)
    if pid >= n_channels:
        return
    
    # Compute output index for this channel
    output_idx = pid
    
    # Compute global input base for this batch and channel
    spatial_elements = height * width
    
    # Sum all spatial elements across batches for this channel
    channel_sum = 0.0
    count = 0
    
    for batch in range(n_batches):
        for h in range(height):
            for w in range(width):
                # Compute input offset
                input_offset = batch * n_channels * spatial_elements + pid * spatial_elements + h * width + w
                val = tl.load(input_ptr + input_offset, other=0.0)
                channel_sum += val
                count += 1
    
    # Compute mean
    if count > 0:
        mean_val = channel_sum / count
    else:
        mean_val = 0.0
    
    # Store the result
    tl.store(output_ptr + output_idx, mean_val)

# Kernel wrapper for optimized mean computation
@torch.fx.wrap
def optimized_mean_spatial(input_tensor):
    # Get tensor dimensions
    shape = input_tensor.shape
    n_batches, n_channels, height, width = shape
    
    # Output will be [n_batches, n_channels]
    output_shape = (n_batches, n_channels)
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    BLOCK_SIZE = 1024  # This will be used per channel processing
    num_channels = n_channels
    
    # Launch kernel for each channel
    optimized_mean_kernel[(num_channels,)](
        input_tensor,
        output,
        n_channels,
        height,
        width,
        n_batches,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

# Replacement function (no arguments, returns function reference)
def replacement_func():
    return optimized_mean_spatial