import torch
import triton
import triton.language as tl
import math

def pattern(x):
    # Match the complete pattern: concat + cos + sin to enable optimization
    concatenated = torch.cat((x, x), dim=-1)
    cos_result = concatenated.cos()
    sin_result = concatenated.sin()
    return cos_result, sin_result

def replacement_args(x):
    return (x,)

@triton.jit
def optimized_trig_kernel(
    x_ptr,
    cos_out_ptr,
    sin_out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data once
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute both trig functions efficiently
    cos_val = tl.cos(x)
    sin_val = tl.sin(x)
    
    # Store results twice (to simulate concatenation without the expensive input concat)
    # This avoids the expensive torch.concat operation on the input
    # cos_out will be [cos_val, cos_val]
    # sin_out will be [sin_val, sin_val]
    
    # For the first half (original positions)
    cos_mask = mask
    sin_mask = mask
    tl.store(cos_out_ptr + offsets, cos_val, mask=cos_mask)
    tl.store(sin_out_ptr + offsets, sin_val, mask=sin_mask)
    
    # For the second half (concatenated positions) - use same values
    # We need to calculate concatenated offsets
    if BLOCK_SIZE > 1:
        concat_offsets = offsets + n_elements
        concat_mask = concat_offsets < (2 * n_elements)
        tl.store(cos_out_ptr + concat_offsets, cos_val, mask=concat_mask)
        tl.store(sin_out_ptr + concat_offsets, sin_val, mask=concat_mask)

@torch.fx.wrap
def optimized_trigonometric_operations(x):
    # Get input tensor info
    original_shape = x.shape
    N = x.numel()
    concatenated_N = 2 * N  # We want output to be 2x the size
    
    # Optimal block size for GPU
    BLOCK_SIZE = 1024
    num_programs = (concatenated_N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensors that are twice as large as input
    # This simulates the effect of torch.cat((x, x), dim=-1) without the expensive operation
    cos_out = torch.empty((concatenated_N,) + original_shape[1:], dtype=x.dtype, device=x.device)
    sin_out = torch.empty((concatenated_N,) + original_shape[1:], dtype=x.dtype, device=x.device)
    
    # Launch the optimized kernel
    optimized_trig_kernel[(num_programs,)](
        x_ptr=x,
        cos_out_ptr=cos_out,
        sin_out_ptr=sin_out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Return both results as (cos_concatenated, sin_concatenated)
    # This matches the expected pattern return signature
    return cos_out, sin_out

def replacement_func():
    return optimized_trigonometric_operations