import torch
import triton
import triton.language as tl

# Pattern matching function - Case without keepdim
def pattern(in_0):
    """Match SILU + Mean reduction pattern without keepdim"""
    # SILU: x * sigmoid(x) - direct implementation
    tmp_sigmoid = torch.sigmoid(in_0)
    tmp_0 = in_0 * tmp_sigmoid
    tmp_1 = tmp_0.mean((2, 3))
    return (tmp_1, tmp_0)

# Argument extraction function  
def replacement_args(in_0):
    """Extract input tensor"""
    return (in_0,)

# Custom Triton kernel that fuses SILU and mean reduction
@triton.jit
def silu_mean_kernel(
    x_ptr,
    silu_out_ptr,
    mean_out_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_CHANNELS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused SILU + Mean reduction kernel for no-keepdim case"""
    pid_batch = tl.program_id(0)
    pid_channel = tl.program_id(1)
    
    # Calculate input tensor position
    x_offset = pid_batch * (channels * height * width) + pid_channel * (height * width)
    
    # Load input block
    x_base_ptr = x_ptr + x_offset
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < height * width
    
    x = tl.load(x_base_ptr + offsets, mask=mask, other=0.0)
    
    # Compute SILU: x * sigmoid(x)
    x_sigmoid = 1.0 / (1.0 + tl.exp(-x))
    silu_out = x * x_sigmoid
    
    # Store SILU output  
    silu_base_ptr = silu_out_ptr + x_offset
    tl.store(silu_base_ptr + offsets, silu_out, mask=mask)
    
    # Compute mean reduction over spatial dimensions
    local_sum = tl.sum(silu_out, axis=0)
    total_elements = height * width
    mean_val = local_sum / total_elements
    
    # Store mean result (no keepdim - 2D tensor)
    mean_out_base_ptr = mean_out_ptr + (pid_batch * channels + pid_channel)
    tl.store(mean_out_base_ptr, mean_val)

# Core fused SILU + Mean reduction implementation
def _fused_silu_mean_kernel_no_keepdim(x):
    """Fused SILU + Mean reduction operation for no-keepdim case"""
    if x.dim() != 4:
        raise ValueError("Input must be 4D tensor [B, C, H, W]")
    
    batch_size, channels, height, width = x.shape
    
    # Allocate output tensors
    silu_out = torch.empty_like(x)
    mean_out = torch.empty(batch_size, channels, device=x.device, dtype=x.dtype)
    
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
    silu_mean_kernel[grid](
        x,
        silu_out,
        mean_out,
        batch_size,
        channels, 
        height,
        width,
        BLOCK_CHANNELS,
        BLOCK_SIZE
    )
    
    # Return in the order expected by pattern: (mean, silu)
    return mean_out, silu_out

# Wrapper function 
@torch.fx.wrap
def fused_silu_mean_no_keepdim(x):
    """Fused SILU + Mean reduction for no-keepdim case"""
    return _fused_silu_mean_kernel_no_keepdim(x)

# Replacement function (returns function reference)
def replacement_func():
    return fused_silu_mean_no_keepdim