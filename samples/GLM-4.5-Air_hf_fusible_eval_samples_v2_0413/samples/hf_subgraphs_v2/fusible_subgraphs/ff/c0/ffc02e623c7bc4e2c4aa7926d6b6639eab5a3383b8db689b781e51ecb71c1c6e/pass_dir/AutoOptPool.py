import torch
import triton
import triton.language as tl

def pattern(x):
    tmp_6 = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
    return tmp_6

def replacement_args(x):
    return (x,)

@triton.jit
def simple_pool2d_kernel(
    input_ptr,
    output_ptr,
    n_batches: tl.constexpr,
    n_channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
):
    # Each program handles one channel batch combination
    pid = tl.program_id(0)
    
    # Calculate batch and channel indices
    batch_idx = pid // n_channels
    ch_idx = pid % n_channels
    
    if batch_idx >= n_batches or ch_idx >= n_channels:
        return
    
    # Compute spatial average
    spatial_sum = 0.0
    spatial_count = height * width
    
    for h in range(height):
        for w in range(width):
            input_offset = (batch_idx * n_channels + ch_idx) * height * width + h * width + w
            val = tl.load(input_ptr + input_offset)
            spatial_sum += val
    
    # Store the average
    output_offset = (batch_idx * n_channels + ch_idx) * 1 * 1 + 0 * 1 + 0
    tl.store(output_ptr + output_offset, spatial_sum / spatial_count)

@torch.fx.wrap
def simple_pool2d(x):
    n_batches, n_channels, height, width = x.shape
    output = torch.empty((n_batches, n_channels, 1, 1), dtype=x.dtype, device=x.device)
    
    n_programs = n_batches * n_channels
    simple_pool2d_kernel[(n_programs,)](
        input_ptr=x,
        output_ptr=output,
        n_batches=n_batches,
        n_channels=n_channels,
        height=height,
        width=width,
    )
    
    return output

def replacement_func():
    return simple_pool2d