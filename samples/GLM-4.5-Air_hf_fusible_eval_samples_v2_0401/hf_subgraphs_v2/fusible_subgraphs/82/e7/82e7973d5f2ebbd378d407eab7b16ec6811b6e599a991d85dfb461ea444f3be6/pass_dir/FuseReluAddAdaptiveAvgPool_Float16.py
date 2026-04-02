import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(x, y):
    """Match ReLU + Add + Adaptive Avg Pool pattern"""
    tmp_0 = torch.nn.functional.relu(y, inplace = False);  y = None
    tmp_1 = tmp_0 + x;  tmp_0 = x = None
    tmp_2 = torch.nn.functional.adaptive_avg_pool2d(tmp_1, 1);  tmp_1 = None
    return (tmp_2,)

# Argument extraction function
def replacement_args(x, y):
    return (x, y)

# @triton.jit
@triton.jit
def relu_add_adaptive_avg_pool_kernel_fp16(
    x_ptr,
    y_ptr,
    out_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused ReLU + Add + Adaptive Avg Pool kernel for float16"""
    
    pid = tl.program_id(0)
    batch_id = pid // channels
    channel_id = pid % channels
    
    if batch_id >= batch_size or channel_id >= channels:
        return
    
    # Load input elements for this channel
    x_offset = batch_id * channels * height * width + channel_id * height * width
    y_offset = x_offset
    
    # Load x and y slices for this channel
    x_block = tl.load(x_ptr + x_offset + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < height * width, other=0.0).to(tl.float32)
    y_block = tl.load(y_ptr + y_offset + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < height * width, other=0.0).to(tl.float32)
    
    # Compute ReLU on y and add x
    relu_y = tl.maximum(y_block, 0.0)
    sum_result = relu_y + x_block
    
    # Adaptive average pool to 1x1 - compute mean of entire spatial dimension
    spatial_size = height * width
    mean_val = tl.sum(sum_result) / spatial_size
    
    # Store result (output is [batch_size, channels, 1, 1])
    out_offset = batch_id * channels + channel_id
    tl.store(out_ptr + out_offset, mean_val.to(tl.float16))

@torch.fx.wrap
def fused_relu_add_adaptive_avg_pool_fp16(x, y):
    """Fused ReLU + Add + Adaptive Avg Pool wrapper for float16"""
    
    # Get tensor shapes
    batch_size, channels, height, width = x.shape
    
    # Allocate output tensor [batch_size, channels, 1, 1]
    output = torch.empty((batch_size, channels, 1, 1), dtype=torch.float16, device=x.device)
    
    # Flatten for processing
    x_flat = x.flatten()
    y_flat = y.flatten()
    output_flat = output.flatten()
    
    # BLOCK_SIZE should match spatial size for this case
    BLOCK_SIZE = height * width
    
    # Calculate grid size: batch_size * channels
    grid_size = batch_size * channels
    
    # Launch kernel
    relu_add_adaptive_avg_pool_kernel_fp16[(grid_size,)](
        x_ptr=x_flat,
        y_ptr=y_flat,
        out_ptr=output_flat,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Replacement function
def replacement_func():
    return fused_relu_add_adaptive_avg_pool_fp16