import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0, in_1):
    tmp_0 = torch.cat([in_0, in_1], dim = 1)
    tmp_1 = tmp_0[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, None))]
    tmp_2 = tmp_1.mean((2, 3), keepdim = True)
    return (tmp_1, tmp_2)

# Argument extraction function
def replacement_args(in_0, in_1):
    tmp_0 = torch.cat([in_0, in_1], dim=1)
    tmp_1 = tmp_0[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, None))]
    return (tmp_1,)

# Optimized kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=2, num_warps=4),
    ],
    key=['height', 'width']
)
@triton.jit
def mean_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    tid = tl.thread_id(0)
    block_idx = tl.program_id(0)
    
    batch_idx = block_idx // channels
    channel_idx = block_idx % channels
    
    output_idx = batch_idx * channels + channel_idx
    
    acc = 0.0
    spatial_idx = tid
    
    if spatial_idx < height * width:
        input_idx = batch_idx * (channels * height * width) + \
                    channel_idx * (height * width) + spatial_idx
        acc = tl.load(input_ptr + input_idx)
    
    total = tl.sum(acc)
    mean_val = total / (height * width)
    
    tl.store(output_ptr + output_idx, mean_val)

# Kernel wrapper
@torch.fx.wrap
def mean_wrapper(input_tensor):
    batch_size, channels, height, width = input_tensor.shape
    output = torch.empty(batch_size, channels, 1, 1, device=input_tensor.device, dtype=input_tensor.dtype)
    
    grid_dim = batch_size * channels
    mean_kernel[(grid_dim, 1)](
        input_ptr=input_tensor,
        output_ptr=output,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE=128
    )
    
    return output

# Replacement function
def replacement_func():
    return mean_wrapper