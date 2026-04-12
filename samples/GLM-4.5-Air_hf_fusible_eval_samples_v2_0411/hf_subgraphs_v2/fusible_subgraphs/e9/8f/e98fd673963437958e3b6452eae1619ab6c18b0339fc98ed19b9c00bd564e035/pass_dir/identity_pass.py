import torch
import triton
import triton.language as tl

def pattern(a, b):
    result = a * b  # This matches the multiplication operation in the graphs
    return result

def replacement_args(a, b):
    return (a, b)

@triton.jit
def multiply_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    out = a * b
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def multiply_function(a, b):
    N = a.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(a)
    
    multiply_kernel[(num_programs,)](
        a_ptr=a,
        b_ptr=b,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return multiply_function