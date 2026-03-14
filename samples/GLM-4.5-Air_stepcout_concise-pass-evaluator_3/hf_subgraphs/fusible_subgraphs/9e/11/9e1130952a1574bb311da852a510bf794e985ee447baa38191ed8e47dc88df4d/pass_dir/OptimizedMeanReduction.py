import torch
import triton
import triton.language as tl

def pattern(x):
    """Pattern to match mean reduction over spatial dimensions"""
    result = x.mean((2, 3), keepdim=False)
    return result

def replacement_args(x):
    """Arguments needed for the replacement"""
    return (x,)

@triton.jit
def optimized_mean_kernel_2d(
    x_ptr,
    out_ptr,
    batch_size,
    n_channels, 
    spatial_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized mean reduction kernel with better memory access"""
    # Each program handles one element in the output [batch_size, n_channels]
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    
    if batch_idx >= batch_size or channel_idx >= n_channels:
        return
    
    # Load all elements for this batch and channel across spatial dimensions
    spatial_loaded = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    count = 0
    
    # Iterate through spatial dimensions
    for i in range(0, spatial_size, BLOCK_SIZE):
        remaining = min(BLOCK_SIZE, spatial_size - i)
        
        # Create offset mask for this iteration
        offsets = i + tl.arange(0, remaining)
        linear_indices = (batch_idx * n_channels + channel_idx) * spatial_size + offsets
        
        # Load spatial elements
        spatial_data = tl.load(x_ptr + linear_indices, mask=offsets < spatial_size, other=0.0)
        spatial_loaded[:remaining] = spatial_data
        count += tl.sum(spatial_data != 0.0)
    
    # Compute mean using vectorized sum
    if count > 0:
        mean_val = tl.sum(spatial_loaded) / spatial_size
    else:
        mean_val = 0.0
    
    # Store result
    output_idx = batch_idx * n_channels + channel_idx
    tl.store(out_ptr + output_idx, mean_val)

@torch.fx.wrap
def optimized_mean_torch(x):
    """Optimized mean reduction with better Triton kernel"""
    batch_size, n_channels, height, width = x.shape
    spatial_size = height * width
    
    # Output: [batch_size, n_channels]
    out = torch.empty(batch_size, n_channels, dtype=x.dtype, device=x.device)
    
    # Launch kernel  
    grid = (batch_size, n_channels)
    optimized_mean_kernel_2d[grid](
        x_ptr=x,
        out_ptr=out,
        batch_size=batch_size,
        n_channels=n_channels,
        spatial_size=spatial_size,
        BLOCK_SIZE=1024
    )
    
    return out

def replacement_func():
    """Return the optimized mean function"""
    return optimized_mean_torch