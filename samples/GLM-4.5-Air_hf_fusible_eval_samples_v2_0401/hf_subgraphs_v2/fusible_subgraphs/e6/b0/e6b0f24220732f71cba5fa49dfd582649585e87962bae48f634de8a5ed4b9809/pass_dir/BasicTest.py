import torch
import triton
import triton.language as tl

def pattern(x, y):
    """Simple pattern like the reference example"""
    return x + y

def replacement_args(x, y):
    return (x, y)

@triton.jit
def basic_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Basic addition kernel"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def basic_add(x, y):
    """Basic addition function"""
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    basic_add_kernel[(num_programs,)](
        x, y, out, N, BLOCK_SIZE
    )
    
    return out

def replacement_func():
    return basic_add