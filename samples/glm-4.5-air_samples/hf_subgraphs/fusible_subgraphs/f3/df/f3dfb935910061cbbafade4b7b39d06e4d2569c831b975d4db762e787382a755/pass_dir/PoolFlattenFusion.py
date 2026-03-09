import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(x_in):
    tmp_2 = torch.nn.functional.adaptive_avg_pool2d(x_in, 1)
    tmp_3 = tmp_2.flatten(1, -1)
    return (tmp_3,)

# Argument extraction function
def replacement_args(x):
    return (x,)

# Optimized kernel for adaptive avg pool2d (output size 1) + flatten fusion
@triton.jit
def pool_flatten_kernel(x_ptr, out_ptr, n_batch, n_channels, height, width, BATCH_SIZE: tl.constexpr, CHANNEL_SIZE: tl.constexpr):
    # Program ID determines which batch and channel to process
    pid = tl.program_id(0)
    
    # Split program ID into batch and channel indices
    batch_idx = pid // n_channels
    channel_idx = pid % n_channels
    
    # Compute spatial dimensions
    spatial_size = height * width
    
    # Calculate pointer offsets for the input spatial data
    spatial_offset = batch_idx * n_channels * spatial_size + channel_idx * spatial_size
    
    # Load spatial data for this specific batch and channel
    spatial_offsets = spatial_offset + tl.arange(0, spatial_size)
    mask = spatial_offsets < ((batch_idx + 1) * n_channels * spatial_size)
    
    x_channel_data = tl.load(x_ptr + spatial_offsets, mask=mask, other=0.0)
    
    # Adaptive avg pool to 1x1 (compute mean across spatial dimensions)
    channel_mean = tl.sum(x_channel_data) / spatial_size
    
    # Store result in flattened output (batch x channels)
    output_offset = batch_idx * n_channels + channel_idx
    tl.store(out_ptr + output_offset, channel_mean)

@torch.fx.wrap
def fused_pool_flatten(x):
    # Get input dimensions
    n_batch, n_channels, height, width = x.shape
    output_shape = (n_batch, n_channels)
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    
    # Calculate grid size - one program per (batch, channel) pair
    total_programs = n_batch * n_channels
    
    # Launch kernel
    pool_flatten_kernel[(total_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_batch=n_batch,
        n_channels=n_channels,
        height=height,
        width=width,
        BATCH_SIZE=1,
        CHANNEL_SIZE=1
    )
    
    return out

# Replacement function
def replacement_func():
    return fused_pool_flatten