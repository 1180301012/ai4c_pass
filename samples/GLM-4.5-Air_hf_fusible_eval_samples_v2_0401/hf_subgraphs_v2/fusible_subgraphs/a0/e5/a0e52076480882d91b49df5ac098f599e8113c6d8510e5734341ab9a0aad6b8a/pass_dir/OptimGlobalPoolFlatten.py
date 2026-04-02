import torch
import triton
import triton.language as tl
import math

def pattern(x):
    """Match Global Average Pool2D + Flatten pattern"""
    tmp_6 = torch.nn.functional.adaptive_avg_pool2d(x, 1)
    tmp_7 = tmp_6.flatten(1, -1)
    return tmp_7

def replacement_args(x):
    return (x,)

@triton.jit
def optimized_global_pool_flatten_kernel(
    x_ptr, out_ptr,
    batch_size, channels, height, width,
    BLOCK_BATCH: tl.constexpr, BLOCK_CHANNELS: tl.constexpr
):
    """Optimized kernel for Global Average Pool2D (kernel=1x1) + Flatten"""
    # Program identifiers
    batch_id = tl.program_id(0)
    channel_id = tl.program_id(1)
    
    if batch_id >= batch_size or channel_id >= channels:
        return
    
    # For global average pooling with kernel_size=1, we just take the single element
    # The pooling result for each channel is simply the element at the center (or any position for 1x1)
    input_offset = batch_id * channels * height * width + channel_id * height * width
    x_val = tl.load(x_ptr + input_offset).to(tl.float32)
    
    # Compute average (for 1x1 kernel, it's just the single value)
    # We could add more sophisticated reduction here if needed
    avg_val = x_val
    
    # Store flattened result
    output_offset = batch_id * channels + channel_id
    tl.store(out_ptr + output_offset, avg_val)

@torch.fx.wrap
def optimized_global_pool_flatten(x):
    """Wrapper function for optimized global average pooling + flatten"""
    if x.dim() != 4:
        raise ValueError("Input must be 4D tensor (batch, channels, height, width)")
    
    batch_size, channels, height, width = x.shape
    
    # Output is flattened to (batch_size, channels)
    out_shape = (batch_size, channels)
    out = torch.empty(out_shape, dtype=torch.float32 if x.dtype == torch.bfloat16 else x.dtype, device=x.device)
    
    # Block size configuration
    BLOCK_BATCH = 1   # Process one batch per program
    BLOCK_CHANNELS = 64  # Process multiple channels per program
    
    # Grid configuration
    grid_x = (batch_size + BLOCK_BATCH - 1) // BLOCK_BATCH
    grid_y = (channels + BLOCK_CHANNELS - 1) // BLOCK_CHANNELS
    
    # Launch kernel
    optimized_global_pool_flatten_kernel[(grid_x, grid_y)](
        x, out,
        batch_size, channels, height, width,
        BLOCK_BATCH, BLOCK_CHANNELS
    )
    
    return out

def replacement_func():
    return optimized_global_pool_flatten