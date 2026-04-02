import torch
import triton
import triton.language as tl

def pattern(in_5, in_4):
    # Simple add + mean pattern
    tmp_4 = in_5 + in_4
    tmp_5 = tmp_4.mean((2, 3), keepdim=False)
    return tmp_5

def replacement_args(in_5, in_4):
    return (in_5, in_4)

@triton.jit
def add_mean_kernel(
    in5_ptr, in4_ptr,
    out_ptr,
    batch_size, num_channels, height, width,
    BLOCK_SIZE: tl.constexpr,
):
    # Add + mean fusion kernel
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    
    if batch_idx >= batch_size or channel_idx >= num_channels:
        return
        
    # Compute sum over spatial dimensions
    spatial_sum = 0.0
    
    for h in range(height):
        for w in range(width):
            # Compute linear index for this spatial location
            offset = ((batch_idx * num_channels + channel_idx) * height + h) * width + w
            input5_val = tl.load(in5_ptr + offset, mask=True, other=0.0)
            input4_val = tl.load(in4_ptr + offset, mask=True, other=0.0)
            spatial_sum += input5_val + input4_val
    
    # Compute mean
    spatial_size = height * width
    mean_val = spatial_sum / spatial_size
    
    # Store result
    output_idx = batch_idx * num_channels + channel_idx
    tl.store(out_ptr + output_idx, mean_val)

@torch.fx.wrap
def fused_add_mean(in_5, in_4):
    batch_size, num_channels, height, width = in_5.shape
    
    # Create output tensor
    output = torch.empty((batch_size, num_channels), dtype=in_5.dtype, device=in_5.device)
    
    # Create 2D grid
    grid = (batch_size, num_channels)
    
    # Launch kernel
    add_mean_kernel[grid](
        in5_ptr=in_5,
        in4_ptr=in_4,
        out_ptr=output,
        batch_size=batch_size,
        num_channels=num_channels,
        height=height,
        width=width,
        BLOCK_SIZE=1,
    )
    
    return output

def replacement_func():
    return fused_add_mean