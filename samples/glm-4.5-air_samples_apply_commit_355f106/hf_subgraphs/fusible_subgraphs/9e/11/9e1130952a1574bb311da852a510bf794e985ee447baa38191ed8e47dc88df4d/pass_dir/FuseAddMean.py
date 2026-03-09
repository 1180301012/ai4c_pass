import torch
import triton
import triton.language as tl

def pattern(x, y):
    # Match addition followed by mean computation
    # tmp_4 = in_5 + in_4
    # tmp_5 = tmp_4.mean((2, 3), keepdim=False)
    sum_result = x + y
    mean_result = sum_result.mean((2, 3), keepdim=False)
    return mean_result

def replacement_args(x, y):
    return (x, y)

@triton.jit
def fused_add_mean_kernel(
    x_ptr, y_ptr, out_ptr,
    batch_size, channels, height, width,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel that computes (x + y).mean() over spatial dimensions"""
    pid = tl.program_id(0)
    batch_idx = pid // channels
    channel_idx = pid % channels
    
    # Initialize accumulator for this batch and channel
    batch_channel_sum = 0.0
    spatial_elements = height * width
    
    # Calculate base offset for this batch and channel
    base_offset = (batch_idx * channels + channel_idx) * height * width
    
    # Process all spatial positions
    for h in range(height):
        for w in range(width):
            offset = base_offset + h * width + w
            # Load x and y values
            x_val = tl.load(x_ptr + offset)
            y_val = tl.load(y_ptr + offset)
            # Accumulate sum
            batch_channel_sum += (x_val + y_val)
    
    # Compute mean for this batch and channel
    mean_val = batch_channel_sum / spatial_elements
    
    # Store result (output shape is [batch_size, channels])
    output_offset = pid
    tl.store(out_ptr + output_offset, mean_val)

@torch.fx.wrap
def fused_add_mean(x, y):
    """Fused operation: computes (x + y).mean() over spatial dimensions"""
    batch_size, channels, height, width = x.shape
    
    # Allocate output tensor
    out = torch.empty(batch_size * channels, dtype=x.dtype, device=x.device)
    
    # Calculate grid size
    total_elements = batch_size * channels
    BLOCK_SIZE = 1  # Each program handles one batch-channel pair
    
    # Launch kernel
    fused_add_mean_kernel[(total_elements,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape output to [batch_size, channels]
    return out.view(batch_size, channels)

def replacement_func():
    return fused_add_mean