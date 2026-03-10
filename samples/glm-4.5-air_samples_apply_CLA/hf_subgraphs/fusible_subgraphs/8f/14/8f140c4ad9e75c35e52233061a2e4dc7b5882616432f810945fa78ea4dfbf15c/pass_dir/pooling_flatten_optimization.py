import torch
import triton
import triton.language as tl
import math

@triton.jit
def adaptive_avg_pool2d_1x1_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Each program handles one output element (one batch, one channel)
    batch_idx = pid // channels
    channel_idx = pid % channels
    
    # Check bounds
    if batch_idx >= batch_size or channel_idx >= channels:
        return
    
    # Compute the sum of all spatial elements for this batch and channel
    spatial_elements = height * width
    spatial_sum = 0.0
    
    # Iterate over spatial dimensions (small loop, should be efficient)
    for h in range(height):
        for w in range(width):
            # Calculate input index for current spatial location
            input_idx = ((batch_idx * channels + channel_idx) * height + h) * width + w
            # Load and accumulate
            spatial_sum += tl.load(input_ptr + input_idx)
    
    # Compute average
    avg_val = spatial_sum / spatial_elements
    
    # Store result
    output_idx = batch_idx * channels + channel_idx
    tl.store(output_ptr + output_idx, avg_val)

@torch.fx.wrap
def adaptive_avg_pool2d_1x1_optimized(input_tensor):
    batch_size, channels, height, width = input_tensor.shape
    
    # Output is [batch_size, channels] (effectively after flattening)
    output = torch.empty((batch_size, channels), device=input_tensor.device, dtype=input_tensor.dtype)
    
    # Calculate grid configuration - one program per output element
    total_programs = batch_size * channels
    grid = (total_programs,)
    
    # Launch kernel
    adaptive_avg_pool2d_1x1_kernel[grid](
        input_tensor,
        output,
        batch_size,
        channels,
        height,
        width,
        1024  # BLOCK_SIZE is not used in the new kernel but required by signature
    )
    
    return output

def pattern(input_tensor):
    # Match adaptive_avg_pool2d + flatten pattern
    # tmp_9 = torch.nn.functional.adaptive_avg_pool2d(tmp_8, 1)
    # tmp_10 = tmp_9.flatten(1, -1)
    pooled = torch.nn.functional.adaptive_avg_pool2d(input_tensor, 1)
    flattened = pooled.flatten(1, -1)
    return flattened

def replacement_args(input_tensor):
    return (input_tensor,)

def replacement_func():
    return adaptive_avg_pool2d_1x1_optimized