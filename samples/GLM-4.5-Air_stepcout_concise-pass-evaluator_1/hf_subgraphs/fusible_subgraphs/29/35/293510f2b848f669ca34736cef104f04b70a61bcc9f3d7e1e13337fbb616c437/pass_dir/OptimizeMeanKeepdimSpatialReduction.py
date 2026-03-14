import torch
import triton
import triton.language as tl

# Pattern matching function - Match mean operation with keepdim=True
def pattern(silu_out):
    """Match spatial mean reduction pattern with keepdim=True"""
    mean_result = silu_out.mean((2, 3), keepdim=True)
    return mean_result

# Argument extraction function  
def replacement_args(silu_out):
    """Extract the silu output tensor"""
    return (silu_out,)

# Custom Triton kernel for optimized mean reduction with keepdim
@triton.jit
def mean_keepdim_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized mean reduction kernel with keepdim=True"""
    pid_batch = tl.program_id(0)
    pid_channel = tl.program_id(1)
    
    # Each thread loads a block of spatial data
    offsets = tl.arange(0, BLOCK_SIZE)
    spatial_elements = height * width
    mask = offsets < spatial_elements
    
    # Calculate base position for this batch+channel
    base_pos = pid_batch * (channels * spatial_elements) + pid_channel * spatial_elements
    
    # Load spatial block
    x_block = tl.load(x_ptr + base_pos + offsets, mask=mask, other=0.0)
    
    # Sum all spatial elements
    channel_sum = tl.sum(x_block)
    total_elements = tl.sum(mask.to(tl.float32))
    mean_val = channel_sum / total_elements
    
    # Store mean result in [B, C, 1, 1] layout - only one thread per (batch, channel) stores
    # Use pid_offset to identify which thread (0, 1, 2, ...) this is
    pid_offset = tl.program_id(2)
    if pid_offset == 0:  # Only first thread per (batch, channel) pair stores the result
        out_idx = (pid_batch * channels + pid_channel) * 4
        tl.store(out_ptr + out_idx + 0, mean_val)
        tl.store(out_ptr + out_idx + 1, mean_val)
        tl.store(out_ptr + out_idx + 2, mean_val)
        tl.store(out_ptr + out_idx + 3, mean_val)

# Core mean reduction implementation with keepdim
def _optimized_mean_keepdim(x):
    """Optimized mean reduction operation with keepdim=True"""
    if x.dim() != 4:
        raise ValueError("Input must be 4D tensor [B, C, H, W]")
    
    batch_size, channels, height, width = x.shape
    
    # Allocate output tensor with keepdim shape [B, C, 1, 1]
    out = torch.empty(batch_size, channels, 1, 1, device=x.device, dtype=x.dtype)
    
    # Use BLOCK_SIZE large enough to cover all spatial locations efficiently
    spatial_elements = height * width
    BLOCK_SIZE = min(1024, spatial_elements)  # Use 1024 or smaller for small tensors
    
    # Calculate grid dimensions: (batch, channel, num_blocks_per_channel)
    grid = (batch_size, channels, 1)  # Use single block per (batch, channel)
    
    # Launch kernel
    mean_keepdim_kernel[grid](
        x,
        out,
        batch_size,
        channels, 
        height,
        width,
        BLOCK_SIZE
    )
    
    return out

# Wrapper function 
@torch.fx.wrap
def optimized_mean_keepdim_reduction(x):
    """Optimized mean reduction with keepdim=True"""
    return _optimized_mean_keepdim(x)

# Replacement function (returns function reference)
def replacement_func():
    return optimized_mean_keepdim_reduction