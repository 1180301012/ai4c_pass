import torch
import triton
import triton.language as tl

def pattern(a, b):
    # Simple addition pattern
    return a + b

def replacement_args(a, b):
    return (a, b)

@triton.jit
def simple_add_kernel(
    a_ptr, b_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    
    # Calculate
    out = a + b
    
    # Store
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_simple_add(a, b):
    n_elements = a.numel()
    out = torch.empty_like(a)
    
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    simple_add_kernel[(num_programs,)](
        a_ptr=a,
        b_ptr=b,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_simple_add