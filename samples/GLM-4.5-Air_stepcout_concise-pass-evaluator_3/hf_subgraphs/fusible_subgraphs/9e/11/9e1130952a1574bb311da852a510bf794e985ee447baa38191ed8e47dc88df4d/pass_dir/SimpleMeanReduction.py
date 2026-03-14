import torch
import triton
import triton.language as tl

def pattern(x):
    """Pattern to match mean reduction over spatial dimensions"""
    # Simple pattern that matches just the mean operation
    result = x.mean((2, 3), keepdim=False)
    return result

def replacement_args(x):
    """Arguments needed for the replacement"""
    return (x,)

@triton.jit
def simple_mean_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    n_channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    """Simple mean reduction kernel"""
    # Each program handles one channel
    channel_idx = tl.program_id(0)
    
    if channel_idx >= n_channels:
        return
    
    # Compute mean over spatial dimensions for each batch element
    spatial_elements = height * width
    
    for batch_idx in range(batch_size):
        # Initialize sum for this batch and channel
        sum_val = 0.0
        
        # Iterate through spatial dimensions
        for h_idx in range(0, height, BLOCK_SIZE):
            for w_idx in range(0, width, BLOCK_SIZE):
                # Calculate linear index for current position
                linear_idx = (batch_idx * n_channels + channel_idx) * height * width + h_idx * width + w_idx
                
                # Calculate range for this iteration
                h_end = min(h_idx + BLOCK_SIZE, height)
                w_end = min(w_idx + BLOCK_SIZE, width)
                
                for h in range(h_idx, h_end):
                    for w in range(w_idx, w_end):
                        # Calculate linear index for current position
                        linear_idx = (batch_idx * n_channels + channel_idx) * height * width + h * width + w
                        
                        # Load one element at a time
                        val = tl.load(x_ptr + linear_idx).to(tl.float32)
                        sum_val += val
        
        # Compute mean
        mean_val = sum_val / spatial_elements
        
        # Store result
        result_offset = batch_idx * n_channels + channel_idx
        tl.store(out_ptr + result_offset, mean_val)

@torch.fx.wrap
def simple_optimized_mean(x):
    """Simple optimized mean reduction"""
    batch_size, n_channels, height, width = x.shape
    
    # Output: [batch_size, n_channels]
    out = torch.empty(batch_size, n_channels, dtype=x.dtype, device=x.device)
    
    # Launch kernel
    grid = (n_channels,)
    simple_mean_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        batch_size=batch_size,
        n_channels=n_channels,
        height=height,
        width=width,
        BLOCK_SIZE=1024
    )
    
    return out

def replacement_func():
    """Return the optimized mean function"""
    return simple_optimized_mean