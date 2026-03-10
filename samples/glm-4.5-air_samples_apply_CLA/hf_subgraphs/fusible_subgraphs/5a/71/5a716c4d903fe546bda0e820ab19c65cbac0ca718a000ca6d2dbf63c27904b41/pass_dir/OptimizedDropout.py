import torch
import triton
import triton.language as tl

def pattern(x):
    # Pattern: dropout with fixed parameters (training=False, inplace=False) 
    # Since training=False, dropout is just scaling by 0.9
    out = x * 0.9
    return (out,)

def replacement_args(x):
    return (x,)

@triton.jit
def simple_scale_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    scale_factor: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Apply scaling (equivalent to dropout with training=False)
    out = x * scale_factor
    
    # Store result
    tl.store(output_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_dropout(x):
    # For inference, dropout with training=False is just scaling by 0.9
    scale_factor = 0.9
    
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.empty_like(x)
    
    simple_scale_kernel[(num_programs,)](
        x,
        output,
        N,
        scale_factor=scale_factor,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return optimized_dropout