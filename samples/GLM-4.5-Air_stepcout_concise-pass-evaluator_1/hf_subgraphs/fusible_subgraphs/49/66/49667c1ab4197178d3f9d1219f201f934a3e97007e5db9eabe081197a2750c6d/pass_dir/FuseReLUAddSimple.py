import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """Match ReLU + Addition pattern"""
    tmp_0 = torch.nn.functional.relu(in_1, inplace=False)
    tmp_1 = tmp_0 + in_0
    return tmp_1

def replacement_args(in_0, in_1):
    """Extract arguments for fused ReLU+Addition kernel"""
    return (in_0, in_1)

@triton.jit
def fused_relu_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused ReLU + Addition kernel optimized for spatial processing"""
    total_elements = batch_size * channels * height * width
    pid = tl.program_id(0)
    
    # Calculate element offsets for this program
    start_idx = pid * BLOCK_SIZE
    end_idx = min(start_idx + BLOCK_SIZE, total_elements)
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Calculate batch, channel, spatial coordinates
    offset_1d = offsets
    batch_idx = offset_1d // (channels * height * width)
    channel_idx = (offset_1d // (height * width)) % channels
    spatial_idx = offset_1d % (height * width)
    
    # Calculate 2D spatial coordinates
    h = spatial_idx // width
    w = spatial_idx % width
    
    # Calculate flat indices for input tensors
    x_offset = batch_idx * channels * height * width + channel_idx * height * width + spatial_idx
    y_offset = batch_idx * channels * height * width + channel_idx * height * width + spatial_idx
    
    # Load inputs
    x_val = tl.load(x_ptr + x_offset, mask=mask, other=0.0)
    y_val = tl.load(y_ptr + y_offset, mask=mask, other=0.0)
    
    # Fused operation: ReLU(y) + x
    relu_y = tl.maximum(y_val, 0.0)
    out_val = relu_y + x_val
    
    # Store result
    tl.store(out_ptr + offsets, out_val, mask=mask)

@torch.fx.wrap
def fused_relu_add(x, y):
    """Fused ReLU + Addition operation"""
    batch_size, channels, height, width = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
    total_elements = batch_size * channels * height * width
    
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    fused_relu_add_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    """Return the fused ReLU+Addition function"""
    return fused_relu_add