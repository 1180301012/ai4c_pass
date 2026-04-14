import torch
import triton
import triton.language as tl

def pattern(tmp_4):
    # Simple pattern: mean reduction operation
    tmp_5 = tmp_4.mean((2, 3), keepdim=False)
    return tmp_5

def replacement_args(tmp_4):
    # Extract argument for optimized mean reduction
    return (tmp_4,)

@triton.jit
def optimized_mean_kernel(
    input_ptr,
    output_ptr,
    n_batch: tl.constexpr,
    n_channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
):
    """Optimized mean reduction kernel"""
    pid = tl.program_id(0)
    
    batch_idx = pid // n_channels
    channel_idx = pid % n_channels
    
    if batch_idx >= n_batch or channel_idx >= n_channels:
        return
    
    # Compute mean for this batch and channel
    input_offset = batch_idx * n_channels * height * width + channel_idx * height * width
    spatial_sum = 0.0
    
    for i in range(height):
        for j in range(width):
            offset = input_offset + i * width + j
            val = tl.load(input_ptr + offset)
            spatial_sum += val
    
    mean_val = spatial_sum / (height * width)
    
    # Store result at batch_idx * n_channels + channel_idx
    output_offset = batch_idx * n_channels + channel_idx
    tl.store(output_ptr + output_offset, mean_val)

@torch.fx.wrap
def optimized_mean(input_tensor):
    """Optimized mean reduction over spatial dimensions"""
    N, C, H, W = input_tensor.shape
    
    # Create output tensor
    output = torch.empty((N, C), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch kernel
    total_elements = N * C
    num_programs = total_elements
    
    optimized_mean_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=output,
        n_batch=N,
        n_channels=C,
        height=H,
        width=W,
    )
    
    return output

def replacement_func():
    return optimized_mean