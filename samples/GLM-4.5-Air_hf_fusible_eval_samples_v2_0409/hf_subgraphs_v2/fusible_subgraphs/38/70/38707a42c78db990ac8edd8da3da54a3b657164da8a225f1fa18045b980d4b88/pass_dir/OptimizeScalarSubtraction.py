import torch
import triton
import triton.language as tl

# Pattern: 5 - x (scalar subtraction)
def pattern(x):
    return 5 - x

def replacement_args(x):
    return (x,)

@triton.jit
def optimized_subtract_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of elements
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for bounds checking
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Subtract from 5
    result = 5 - x
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def optimized_subtract(x):
    n_elements = x.numel()
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Choose block size based on tensor size
    if n_elements <= 1024:
        BLOCK_SIZE = n_elements
    else:
        BLOCK_SIZE = 1024
    
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    optimized_subtract_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_subtract