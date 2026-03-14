import torch
import triton
import triton.language as tl

# Pattern matching function - match just the view operation on the mean output
def pattern(reduced_tensor):
    output = reduced_tensor.view(1, 1, -1)
    return output

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Triton ReLU kernel
@triton.jit
def relu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Mask to ensure we don't go out of bounds
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Apply ReLU: max(x, 0)
    out = tl.maximum(x, 0.0)
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

# Triton mean reduction kernel
@triton.jit
def mean_reduction_kernel(
    x_ptr,      # Input tensor pointer (after ReLU)
    out_ptr,    # Output for channel means
    channels,   # Number of channels
    height,     # Spatial height
    width,      # Spatial width
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one channel
    channel_idx = tl.program_id(0)
    
    # Calculate spatial size and total elements
    spatial_size = height * width
    spatial_block_start = tl.program_id(1) * BLOCK_SIZE
    spatial_offsets = spatial_block_start + tl.arange(0, BLOCK_SIZE)
    spatial_mask = spatial_offsets < spatial_size
    
    # Calculate pointer for this channel
    channel_offset = channel_idx * spatial_size
    input_ptr = x_ptr + channel_offset
    output_ptr = out_ptr + channel_idx
    
    # Load spatial data for this channel
    spatial_data = tl.load(input_ptr + spatial_offsets, mask=spatial_mask, other=0.0)
    
    # Compute mean for this channel
    channel_sum = tl.sum(spatial_data)
    channel_mean = channel_sum / spatial_size
    
    # Store the mean
    tl.store(output_ptr, channel_mean)

@torch.fx.wrap
def optimized_view(x):
    # Just apply view - this is exactly what the pattern does
    return x.view(1, 1, -1)

# Replacement function (must return function reference, not call)
def replacement_func():
    return optimized_view