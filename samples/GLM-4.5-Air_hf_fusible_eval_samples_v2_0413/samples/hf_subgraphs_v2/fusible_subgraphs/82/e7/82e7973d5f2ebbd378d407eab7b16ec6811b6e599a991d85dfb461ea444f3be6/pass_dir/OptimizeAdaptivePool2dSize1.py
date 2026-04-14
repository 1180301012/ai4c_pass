import torch
import triton
import triton.language as tl

def pattern(x):
    """Optimize adaptive_avg_pool2d with output size 1
    Matches: adaptive_avg_pool2d(x, 1) which computes spatial mean
    """
    result = torch.nn.functional.adaptive_avg_pool2d(x, 1)
    return result

def replacement_args(x):
    """Extract arguments for the optimized pooling kernel"""
    return (x,)

@triton.jit
def optimized_spatial_mean_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel for spatial mean computation (adaptive_avg_pool2d with output size 1)"""
    # Each program handles one batch channel combination
    pid = tl.program_id(0)
    batch_idx = pid // channels
    channel_idx = pid % channels
    
    # Calculate total number of elements per channel
    elements_per_channel = height * width
    
    # Process all spatial elements for this batch and channel
    sum_val = 0.0
    num_elements = 0
    
    for i in range(0, elements_per_channel, BLOCK_SIZE):
        offset_within_channel = i
        offsets = offset_within_channel + tl.arange(0, BLOCK_SIZE)
        mask = offsets < elements_per_channel
        
        # Calculate global pointer offset
        global_offset = batch_idx * channels * elements_per_channel + \
                       channel_idx * elements_per_channel + offsets
        
        # Load element - using vectorized loads
        x_vals = tl.load(x_ptr + global_offset, mask=mask, other=0.0)
        
        # Mask count for this iteration
        current_elements = tl.sum(mask.to(tl.float32))
        
        # Accumulate sum and count
        sum_val += tl.sum(x_vals)
        num_elements += current_elements
    
    # Compute mean - handle division by zero
    if num_elements > 0:
        mean_val = sum_val / num_elements
    else:
        mean_val = 0.0
    
    # Store result at position (batch_idx, channel_idx) in output
    out_offset = batch_idx * channels + channel_idx
    tl.store(out_ptr + out_offset, mean_val)

@torch.fx.wrap
def optimized_adaptive_pool2d_size1(x):
    """Wrapper for optimized adaptive avg pool2d with output size 1"""
    batch_size, channels, height, width = x.shape
    
    # Output shape is [batch_size, channels, 1, 1]
    output_shape = (batch_size, channels, 1, 1)
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    
    # Flatten spatial dimensions for computation
    if height * width > 0:
        BLOCK_SIZE = 1024
        num_programs = batch_size * channels
        
        optimized_spatial_mean_kernel[(num_programs,)](
            x_ptr=x,
            out_ptr=out.view(batch_size * channels),  # Flatten output
            batch_size=batch_size,
            channels=channels,
            height=height,
            width=width,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    # Reshape to expected output format [batch_size, channels, 1, 1]
    return out.view(batch_size, channels, 1, 1)

def replacement_func():
    """Return the optimized adaptive pooling function"""
    return optimized_adaptive_pool2d_size1