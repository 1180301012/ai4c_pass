import torch
import triton
import triton.language as tl

def pattern(tmp_1):
    """Match adaptive_avg_pool2d for optimization"""
    tmp_2 = torch.nn.functional.adaptive_avg_pool2d(tmp_1, 1)
    return tmp_2

def replacement_args(tmp_1):
    """Extract arguments for global average pooling kernel"""
    return (tmp_1,)

@triton.jit
def global_avg_pool_kernel(
    input_ptr,
    out_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized Global Average Pooling kernel"""
    # Each program handles one batch/channel pair for better cache efficiency
    batch_channel_id = tl.program_id(0)
    
    # Extract batch and channel IDs
    batch_id = batch_channel_id // channels
    channel_id = batch_channel_id % channels
    
    # Check bounds
    if batch_id >= batch_size or channel_id >= channels:
        return
    
    # Calculate memory offset for this batch/channel pair
    channel_size = height * width
    base_offset = (batch_id * channels + channel_id) * channel_size
    
    # Load all spatial positions efficiently
    spatial_offsets = tl.arange(0, BLOCK_SIZE)
    mask = spatial_offsets < channel_size
    
    input_offsets = base_offset + spatial_offsets
    input_vals = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
    
    # Compute sum and count efficiently
    sum_val = tl.sum(input_vals)
    count = tl.sum(mask.to(tl.float32))
    
    # Compute mean and store
    if count > 0:
        mean_val = sum_val / count
        out_offset = batch_id * channels + channel_id
        tl.store(out_ptr + out_offset, mean_val)

@torch.fx.wrap
def optimized_global_avg_pool(input_tensor):
    """Optimized Global Average Pooling"""
    batch_size, channels, height, width = input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3]
    
    # Choose optimal block size based on spatial dimensions (must be power of 2)
    channel_size = height * width
    BLOCK_SIZE = 64  # Use fixed power of 2 for better performance
    
    # Output shape: [batch_size, channels]
    out = torch.empty((batch_size, channels), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Total number of batch/channel pairs
    total_pairs = batch_size * channels
    
    # 1D grid: one program per batch/channel pair
    num_programs = (total_pairs + 3) // 4  # Round up to nearest multiple of 4 for better GPU utilization
    
    global_avg_pool_kernel[(num_programs,)](
        input_ptr=input_tensor,
        out_ptr=out,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out.view(batch_size, channels, 1, 1)

def replacement_func():
    """Return the optimized global average pooling function"""
    return optimized_global_avg_pool