import torch
import triton
import triton.language as tl

def pattern(x, y):
    tmp_0 = x * y
    tmp_1 = torch.sum(tmp_0, dim=1)
    tmp_2 = tmp_1.unsqueeze(1)
    tmp_3 = torch.sigmoid(tmp_2)
    return tmp_3

def replacement_args(x, y):
    return (x, y)

@triton.jit
def fused_multiply_sum_sigmoid_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    batch_size,
    channels,
    height,
    width,
):
    # Each program handles one spatial position (h, w) and one batch
    h = tl.program_id(0)
    w = tl.program_id(1)
    b = tl.program_id(2)
    
    # Check bounds - use separate conditions to avoid chained operators
    if h >= height:
        return
    if w >= width:
        return
    if b >= batch_size:
        return
    
    # Load and compute for this specific (b, h, w) position across all channels
    x_offset = b * channels * height * width + h * width + w
    y_offset = b * channels * height * width + h * width + w
    
    # Compute sum of element-wise products across channels
    channel_sum = 0.0
    for c in range(channels):
        x_off = x_offset + c * height * width
        y_off = y_offset + c * height * width
        x_val = tl.load(x_ptr + x_off).to(tl.float32)
        y_val = tl.load(y_ptr + y_off).to(tl.float32)
        channel_sum += x_val * y_val
    
    # Apply sigmoid
    result = 1.0 / (1.0 + tl.exp(-channel_sum))
    
    # Store result
    out_offset = b * height * width + h * width + w
    tl.store(out_ptr + out_offset, result)

@torch.fx.wrap
def fused_multiply_sum_sigmoid(x, y):
    # Get input shapes
    batch_size, channels, height, width = x.shape
    
    # Output shape is [batch_size, height, width]
    out_shape = (batch_size, height, width)
    out = torch.empty(out_shape, dtype=torch.float32, device=x.device)
    
    # Calculate grid dimensions - each program handles one (h, w, b) triplet
    grid_h = height
    grid_w = width  
    grid_b = batch_size
    
    # Launch kernel
    fused_multiply_sum_sigmoid_kernel[grid_h, grid_w, grid_b](
        x,
        y,
        out,
        batch_size,
        channels,
        height,
        width
    )
    
    return out.unsqueeze(1)  # Add channel dimension as [B, 1, H, W]

def replacement_func():
    return fused_multiply_sum_sigmoid