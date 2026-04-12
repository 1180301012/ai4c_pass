import torch
import triton
import triton.language as tl

def pattern(x):
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(x, 1)
    tmp_2 = torch.flatten(tmp_1, 1)
    return tmp_2

def replacement_args(x):
    return (x,)

@triton.jit
def global_avg_pool_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute total elements per channel
    total_elements = height * width
    
    # Load all spatial elements for this thread
    spatial_offsets = tl.arange(0, BLOCK_SIZE)
    spatial_mask = spatial_offsets < total_elements
    
    # Load spatial elements for current batch and channel
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    
    x_ptr_batch = x_ptr + batch_idx * channels * height * width
    x_ptr_channel = x_ptr_batch + channel_idx * height * width
    
    # Load spatial elements
    x_vals = tl.load(x_ptr_channel + spatial_offsets, mask=spatial_mask, other=0.0)
    
    # Compute average using multiple steps to avoid overflow
    sum_val = tl.sum(x_vals)
    avg_val = sum_val / total_elements
    
    # Store result
    out_idx = batch_idx * channels + channel_idx
    tl.store(out_ptr + out_idx, avg_val)

@torch.fx.wrap
def fused_global_avg_pool(x):
    batch_size, channels, height, width = x.shape
    
    # Always use optimized Triton kernel (no fallback to forbidden APIs)
    out = torch.empty(batch_size, channels, dtype=x.dtype, device=x.device)
    
    elements_per_channel = height * width
    if elements_per_channel <= 8192:  # Reasonable block size for spatial dimensions
        BLOCK_SIZE = 1024
        while BLOCK_SIZE > elements_per_channel:
            BLOCK_SIZE //= 2
        if BLOCK_SIZE == 0:
            BLOCK_SIZE = 1
    else:
        BLOCK_SIZE = 8192
    
    num_batch = batch_size
    num_channels = channels
    
    global_avg_pool_kernel[(num_batch, num_channels,)](
        x_ptr=x,
        out_ptr=out,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_global_avg_pool