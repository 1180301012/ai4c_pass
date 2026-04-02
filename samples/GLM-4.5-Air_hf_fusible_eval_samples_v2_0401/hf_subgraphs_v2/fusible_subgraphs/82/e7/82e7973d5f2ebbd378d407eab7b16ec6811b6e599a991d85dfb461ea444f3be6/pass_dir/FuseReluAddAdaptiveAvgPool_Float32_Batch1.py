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

@triton.jit
def relu_add_adaptive_avg_pool_kernel_fp32_batch1(
    x_ptr,
    y_ptr,
    out_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused ReLU + Add + Adaptive Avg Pool kernel for float32 batch=1"""
    
    pid = tl.program_id(0)
    
    if pid >= channels:
        return
    
    # Load input elements for this channel (batch=1 case)
    x_offset = pid * height * width
    y_offset = x_offset
    
    # Load x and y slices for this channel
    x_block = tl.load(x_ptr + x_offset + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < height * width, other=0.0)
    y_block = tl.load(y_ptr + y_offset + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < height * width, other=0.0)
    
    # Compute ReLU on y and add x
    relu_y = tl.maximum(y_block, 0.0)
    sum_result = relu_y + x_block
    
    # Adaptive average pool to 1x1 - compute mean of entire spatial dimension
    spatial_size = height * width
    mean_val = tl.sum(sum_result) / spatial_size
    
    # Store result (output is [1, channels, 1, 1] flattened to [channels])
    tl.store(out_ptr + pid, mean_val)

@torch.fx.wrap
def fused_relu_add_adaptive_avg_pool_fp32_batch1(x, y):
    """Fused ReLU + Add + Adaptive Avg Pool wrapper for float32 batch=1"""
    
    # Get tensor shapes
    batch_size, channels, height, width = x.shape
    
    # Allocate output tensor [1, channels, 1, 1]
    output = torch.empty((1, channels, 1, 1), dtype=torch.float32, device=x.device)
    
    # Flatten for processing
    x_flat = x.flatten()
    y_flat = y.flatten()
    output_flat = output.flatten()
    
    # BLOCK_SIZE should match spatial size
    BLOCK_SIZE = height * width
    
    # Calculate grid size: channels (since batch=1)
    grid_size = channels
    
    # Launch kernel
    relu_add_adaptive_avg_pool_kernel_fp32_batch1[(grid_size,)](
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
    return fused_relu_add_adaptive_avg_pool_fp32_batch1