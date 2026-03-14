import torch
import triton
import triton.language as tl

# Pattern matching function - Match just the mean operation after SILU
def pattern(silu_out):
    """Match spatial mean reduction pattern"""
    mean_result = silu_out.mean((2, 3))
    return mean_result

# Argument extraction function  
def replacement_args(silu_out):
    """Extract the silu output tensor"""
    return (silu_out,)

# Custom Triton kernel for optimized mean reduction
@triton.jit
def mean_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_CHANNELS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized mean reduction kernel"""
    pid_batch = tl.program_id(0)
    pid_channel = tl.program_id(1)
    
    # Calculate input tensor position
    x_offset = pid_batch * (channels * height * width) + pid_channel * (height * width)
    
    # Load input block
    x_base_ptr = x_ptr + x_offset
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < height * width
    
    x = tl.load(x_base_ptr + offsets, mask=mask, other=0.0)
    
    # Compute mean reduction over spatial dimensions
    local_sum = tl.sum(x, axis=0)
    total_elements = height * width
    mean_val = local_sum / total_elements
    
    # Store mean result (no keepdim - 2D tensor)
    out_base_ptr = out_ptr + (pid_batch * channels + pid_channel)
    tl.store(out_base_ptr, mean_val)

# Core mean reduction implementation
def _optimized_mean(x):
    """Optimized mean reduction operation"""
    if x.dim() != 4:
        raise ValueError("Input must be 4D tensor [B, C, H, W]")
    
    batch_size, channels, height, width = x.shape
    
    # Allocate output tensor
    out = torch.empty(batch_size, channels, device=x.device, dtype=x.dtype)
    
    # Choose block sizes based on tensor dimensions
    BLOCK_SIZE = min(1024, height * width)
    BLOCK_CHANNELS = 1
    
    # Calculate grid dimensions
    num_batches = batch_size
    num_channels = channels
    num_blocks_per_channel = (height * width + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    if num_blocks_per_channel > 1:
        grid = (num_batches, num_channels, num_blocks_per_channel)
    else:
        grid = (num_batches, num_channels)
    
    # Launch kernel
    mean_kernel[grid](
        x,
        out,
        batch_size,
        channels, 
        height,
        width,
        BLOCK_CHANNELS,
        BLOCK_SIZE
    )
    
    return out

# Wrapper function 
@torch.fx.wrap
def optimized_mean_reduction(x):
    """Optimized mean reduction"""
    return _optimized_mean(x)

# Replacement function (returns function reference)
def replacement_func():
    return optimized_mean_reduction