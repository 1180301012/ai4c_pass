import torch
import triton
import triton.language as tl

def pattern(x, y):
    tmp_0 = x * y
    tmp_1 = torch.sum(tmp_0, dim=1)
    tmp_2 = tmp_1.unsqueeze(1)
    return tmp_2

def replacement_args(x, y):
    return (x, y)

@triton.jit
def fused_sum_unsqueeze_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    batch_size,
    height,
    width,
    channels: tl.constexpr,
):
    # Get 3D program ID
    pid_b = tl.program_id(0)
    pid_y = tl.program_id(1)
    pid_x = tl.program_id(2)
    
    # Check bounds using mask (chained OR not supported in Triton)
    bounds_mask = (pid_b < batch_size) & (pid_y < height) & (pid_x < width)
    if not bounds_mask:
        return
    
    # Initialize accumulator
    sum_val = 0.0
    
    # Calculate total elements per channel
    elements_per_channel = height * width
    
    # Process each channel
    for c in range(channels):
        # Calculate base offset for this batch and channel
        # Shape: [batch, channels, height, width]
        # Memory layout: batch first, then channels, then height, then width
        channel_offset = pid_b * channels * elements_per_channel + c * elements_per_channel
        row_offset = pid_y * width
        element_offset = pid_x
        
        # Calculate final offset
        offset = channel_offset + row_offset + element_offset
        
        # Load x and y values with bounds checking
        mask = True  # We already checked bounds
        x_val = tl.load(x_ptr + offset, mask=mask, other=0.0)
        y_val = tl.load(y_ptr + offset, mask=mask, other=0.0)
        
        # Accumulate the product
        sum_val += x_val * y_val
    
    # Store the result
    # Output shape: [batch, 1, height, width]
    out_offset = pid_b * 1 * height * width + 0 * height * width + pid_y * width + pid_x
    tl.store(out_ptr + out_offset, sum_val)

@torch.fx.wrap  
def fused_sum_unsqueeze(x, y):
    batch_size, channels, height, width = x.shape
    
    # Output shape: [batch, 1, height, width]
    output_shape = (batch_size, 1, height, width)
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    
    # Grid dimensions - one program per output element
    grid = (
        batch_size,
        height,
        width
    )
    
    fused_sum_unsqueeze_kernel[grid](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        batch_size=batch_size,
        height=height,
        width=width,
        channels=channels
    )
    
    return out

def replacement_func():
    return fused_sum_unsqueeze