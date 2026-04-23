import torch

import triton.language as tl
import triton
import triton.language as tl

@triton.jit
def adaptive_avg_pool2d_kernel(
    x_ptr, out_ptr,
    batch_size, channels,
    height, width,
    BLOCK_SIZE: tl.constexpr
):
    # Calculate the current element index (b, c)
    b_idx = tl.program_id(0) // channels
    c_idx = tl.program_id(0) % channels
    
    # Accumulate the sum over height and width
    acc = tl.zeros((), dtype=tl.float32)
    for h in range(height):
        for w in range(width):
            # Calculate the index in the input tensor
            idx = b_idx * channels * height * width + c_idx * height * width + h * width + w
            acc += tl.load(x_ptr + idx)
    
    # Compute the mean
    mean_val = acc / (height * width)
    
    # Store the result in output
    out_idx = b_idx * channels * 1 * 1 + c_idx * 1 * 1
    tl.store(out_ptr + out_idx, mean_val)

@torch.fx.wrap
def adaptive_avg_pool2d_custom(x):
    # Get input shape
    b, c, h, w = x.shape
    # Create output tensor: [b, c, 1, 1]
    out = torch.empty((b, c, 1, 1), dtype=x.dtype, device=x.device)
    
    total_elements = b * c
    BLOCK_SIZE = 128
    num_blocks = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    adaptive_avg_pool2d_kernel[(num_blocks,)](
        x_ptr=x,
        out_ptr=out,
        batch_size=b,
        channels=c,
        height=h,
        width=w,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

def pattern(x):
    return torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))

def replacement_args(x):
    return (x,)

def replacement_func():
    return adaptive_avg_pool2d_custom