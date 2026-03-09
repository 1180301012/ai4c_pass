import torch
import triton
import triton.language as tl

def pattern(x):
    tmp_6 = torch.nn.functional.adaptive_avg_pool2d(x, 1)
    tmp_7 = tmp_6.flatten(1, -1)
    return tmp_7

def replacement_args(x):
    return (x,)

@triton.jit
def spatial_mean_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    channels,
    height,
    width,
    REDUCE_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Each program handles one channel for the entire batch
    channel = pid
    batch_idx = (pid - channel) // channels
    
    # Calculate spatial mean directly
    spatial_total = height * width
    spatial_sum = 0.0
    
    # Load all spatial elements for this channel and batch
    spatial_offset = batch_idx * channels * height * width + channel * height * width
    x_spatial_ptr = x_ptr + spatial_offset
    
    for h in range(height):
        for w in range(width):
            offset = h * width + w
            val = tl.load(x_spatial_ptr + offset)
            spatial_sum += val
    
    # Compute mean
    mean = spatial_sum / spatial_total
    
    # Store result (flattened output)
    out_offset = batch_idx * channels + channel
    tl.store(out_ptr + out_offset, mean)

@torch.fx.wrap
def adaptive_avg_pool2d_flatten_impl(x):
    batch_size, channels, height, width = x.shape
    
    if height == 1 and width == 1:
        # If already 1x1, just flatten
        return x.flatten(1, -1)
    else:
        # Compute spatial mean directly
        n_elements = batch_size * channels
        REDUCE_BLOCK_SIZE = 32
        num_programs = (n_elements + REDUCE_BLOCK_SIZE - 1) // REDUCE_BLOCK_SIZE
        
        out = torch.empty((batch_size, channels), dtype=torch.float32, device=x.device)
        
        spatial_mean_kernel[(num_programs,)](
            x_ptr=x,
            out_ptr=out,
            batch_size=batch_size,
            channels=channels,
            height=height,
            width=width,
            REDUCE_BLOCK_SIZE=REDUCE_BLOCK_SIZE,
        )
        
        return out

def replacement_func():
    return adaptive_avg_pool2d_flatten_impl