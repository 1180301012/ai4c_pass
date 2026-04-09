import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(silu_output):
    """Fuse mean and view operations for better GPU performance"""
    tmp_1 = silu_output.mean((2, 3))
    tmp_4 = tmp_1.view(1, 1, -1)
    return tmp_4

# Argument extraction function
def replacement_args(silu_output):
    return (silu_output,)

# Optimized kernel for fused mean + view
@triton.jit
def fused_mean_view_kernel(
    input_ptr,
    output_ptr,
    n_channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused mean computation and view operation using Triton"""
    # Calculate total number of spatial elements
    spatial_elements = height * width
    
    # Each program handles one channel
    channel_idx = tl.program_id(0)
    
    # Calculate global memory offset for this channel
    input_offset = channel_idx * spatial_elements
    
    # Load all spatial elements for this channel
    offsets = input_offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < spatial_elements
    channel_data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Compute mean using Triton operations
    channel_sum = tl.sum(channel_data)
    channel_mean = channel_sum / spatial_elements
    
    # Store the result (mean value for each channel becomes scalar in output)
    output_pos = channel_idx
    tl.store(output_ptr + output_pos, channel_mean)

@torch.fx.wrap
def fused_mean_view(silu_output):
    """Wrapper for fused mean + view operation"""
    # Get input dimensions
    batch_size, n_channels, height, width = silu_output.shape
    
    # Create output tensor with shape (1, 1, n_channels) equivalent to view(1, 1, -1)
    output_shape = (1, 1, n_channels)
    output = torch.empty(output_shape, dtype=silu_output.dtype, device=silu_output.device)
    
    # Flatten and process each channel separately
    spatial_elements = height * width
    total_elements = n_channels * spatial_elements
    
    # Choose block size
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_mean_view_kernel[(n_channels,)](
        input_ptr=silu_output,
        output_ptr=output,
        n_channels=n_channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Replacement function (returns function reference)
def replacement_func():
    return fused_mean_view