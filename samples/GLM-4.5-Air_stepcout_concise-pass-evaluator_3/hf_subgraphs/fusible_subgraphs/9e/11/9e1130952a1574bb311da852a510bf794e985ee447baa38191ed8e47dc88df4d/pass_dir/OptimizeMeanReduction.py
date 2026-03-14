import torch
import triton
import triton.language as tl

def pattern(x, bn_mean, bn_var, bn_weight, bn_bias):
    """Pattern to match mean reduction followed by no-op operations"""
    # Original sequence:
    # 1. Compute mean over spatial dimensions (2, 3)
    mean_val = x.mean((2, 3), keepdim=False)
    
    # 2. Two no-op dropouts (p=0.0)
    dropout1 = torch.nn.functional.dropout(mean_val, 0.0, False, False)
    dropout2 = torch.nn.functional.dropout(dropout1, 0.0, False, False)
    
    return dropout2

def replacement_args(x, bn_mean, bn_var, bn_weight, bn_bias):
    """Extract arguments needed for optimized mean reduction"""
    return (x,)

@triton.jit
def optimized_mean_kernel(
    x_ptr,
    out_ptr,
    n_channels,
    batch_size,
    spatial_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized mean reduction kernel"""
    # Each program handles one channel
    channel_idx = tl.program_id(0)
    
    if channel_idx >= n_channels:
        return
    
    # Compute starting offset for this channel
    base_offset = channel_idx * batch_size * spatial_size
    
    # Load all elements for this channel across batch and spatial dimensions
    channel_data = tl.load(
        x_ptr + base_offset,
        mask=tl.arange(spatial_size * batch_size) < (spatial_size * batch_size),
        other=0.0
    )
    
    # Compute mean: sum divided by total elements (batch * spatial)
    channel_sum = tl.sum(channel_data)
    total_elements = batch_size * spatial_size
    mean_val = channel_sum / total_elements
    
    # Store the mean value
    tl.store(out_ptr + channel_idx, mean_val)

@torch.fx.wrap
def optimized_mean_reduction(x):
    """Optimized mean reduction using Triton kernel"""
    batch_size, n_channels, height, width = x.shape
    spatial_size = height * width
    total_elements = batch_size * spatial_size
    
    out = torch.empty(n_channels, dtype=x.dtype, device=x.device)
    
    # Launch kernel
    grid = (n_channels,)
    optimized_mean_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        n_channels=n_channels,
        batch_size=batch_size,
        spatial_size=spatial_size,
        BLOCK_SIZE=min(1024, total_elements)
    )
    
    return out

def replacement_func():
    """Return optimized mean reduction function"""
    return optimized_mean_reduction