import torch
import triton
import triton.language as tl

def pattern(x):
    # Match flatten(2) followed by transpose(1, 2)
    tmp_1 = x.flatten(2)
    tmp_2 = tmp_1.transpose(1, 2)
    return tmp_2

def replacement_args(x):
    return (x,)

@triton.jit
def flatten_transpose_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    n_channels,
    spatial_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles an entire batch and channels for one spatial location
    pid = tl.program_id(0)
    
    # Calculate which spatial element we're processing
    spatial_idx = pid
    
    if spatial_idx >= spatial_size:
        return
        
    # Each program handles one spatial position across the entire batch and channels
    offsets_batch = tl.arange(0, batch_size)
    offsets_channel = tl.arange(0, n_channels)
    
    # Create 2D grid of offsets: (batch, channel)
    batch_offsets = offsets_batch[:, None]
    channel_offsets = offsets_channel[None, :]
    
    # Calculate indices in original x tensor: (batch, channel, spatial_idx)
    x_indices = batch_offsets * (n_channels * spatial_size) + channel_offsets * spatial_size + spatial_idx
    
    # Load data
    mask = (batch_offsets < batch_size) & (channel_offsets < n_channels)
    x = tl.load(x_ptr + x_indices, mask=mask, other=0.0)
    
    # Store directly in transposed format: (batch, spatial_idx, channel)
    out_indices = batch_offsets * (spatial_size * n_channels) + spatial_idx * n_channels + channel_offsets
    tl.store(out_ptr + out_indices, x, mask=mask)

@torch.fx.wrap
def flatten_transpose_optimized(x):
    batch_size, n_channels, spatial_size = x.shape
    total_elements = batch_size * n_channels * spatial_size
    
    # Use reasonable block size
    BLOCK_SIZE = 1024
    grid_size = (spatial_size,)
    
    out = torch.empty((batch_size, spatial_size, n_channels), dtype=x.dtype, device=x.device)
    
    # Launch kernel
    flatten_transpose_kernel[grid_size](x_ptr=x, out_ptr=out, batch_size=batch_size, 
                                       n_channels=n_channels, spatial_size=spatial_size)
    
    return out

def replacement_func():
    return flatten_transpose_optimized