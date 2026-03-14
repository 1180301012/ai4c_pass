import torch
import triton
import triton.language as tl

# Simple test pattern - just match add
def pattern(a, b):
    return a + b

def replacement_args(a, b):
    return (a, b)

@triton.jit
def add_kernel(
    a_ptr, b_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    
    out = a + b
    
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def add_wrapper(a, b):
    n_elements = a.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(a)
    
    add_kernel[(num_programs,)](
        a_ptr=a,
        b_ptr=b,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return add_wrapper