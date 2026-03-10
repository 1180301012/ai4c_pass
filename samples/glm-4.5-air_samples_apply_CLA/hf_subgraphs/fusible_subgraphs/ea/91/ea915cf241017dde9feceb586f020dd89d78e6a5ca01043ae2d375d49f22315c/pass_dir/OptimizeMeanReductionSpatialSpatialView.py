import torch
import triton
import triton.language as tl

def pattern(x):
    # Keep the simple pattern that was working before
    return x.mean((2, 3))

def replacement_args(x):
    return (x,)

@triton.jit
def efficient_mean_kernel(
    x_ptr,
    out_ptr,
    n_channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    """More efficient kernel with vectorized memory access"""
    # Each program handles multiple consecutive channels
    channel_idx = tl.program_id(0)
    
    # Handle boundary conditions
    if channel_idx >= n_channels:
        return
    
    # Create offset for this channel
    spatial_size = height * width
    offset = channel_idx * spatial_size
    
    # Create mask for valid elements
    # Each program handles one spatial chunk, so load all elements for this channel
    mask = tl.arange(offset, offset + spatial_size) < (n_channels * spatial_size)
    
    # Load spatial data more efficiently
    # Load all spatial elements for this channel
    vals = tl.load(x_ptr + tl.arange(offset, offset + spatial_size), mask=mask, other=0.0)
    
    # Compute mean using built-in operations that leverage vectorization
    mean_val = tl.sum(vals) / spatial_size
    
    # Store result
    tl.store(out_ptr + channel_idx, mean_val)

@torch.fx.wrap  
def optimized_mean(x):
    """Mean reduction with hybrid approach"""
    spatial_size = x.shape[2] * x.shape[3]
    
    # For very small spatial sizes, use PyTorch (no kernel overhead)
    if spatial_size <= 16:
        return x.mean((2, 3))
    
    n_channels = x.shape[1]
    
    # For small-medium sizes, use simple but efficient Triton kernel
    if spatial_size <= 256:
        output = torch.empty(n_channels, dtype=x.dtype, device=x.device)
        BLOCK_SIZE = 64
        num_programs = (n_channels + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        efficient_mean_kernel[(num_programs,)](
            x_ptr=x,
            out_ptr=output,
            n_channels=n_channels,
            height=x.shape[2],
            width=x.shape[3],
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return output
    
    # For larger sizes, use PyTorch's highly optimized implementation
    return x.mean((2, 3))

def replacement_func():
    return optimized_mean