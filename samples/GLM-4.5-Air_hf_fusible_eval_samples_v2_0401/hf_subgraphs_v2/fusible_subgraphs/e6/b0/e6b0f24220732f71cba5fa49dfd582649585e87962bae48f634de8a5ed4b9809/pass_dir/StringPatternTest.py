import torch
import triton
import triton.language as tl

def pattern(x):
    """Try to match a simple tensor operation"""
    return x * 2.0

def replacement_args(x):
    return (x,)

@triton.jit
def simple_multiply_kernel(
    x_ptr, out_ptr, 
    n_elements, scale, 
    BLOCK_SIZE: tl.constexpr,
):
    """Simple multiplication kernel"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = x * scale
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def simple_multiply(x, scale=2.0):
    """Simple multiplication function"""
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    simple_multiply_kernel[(num_programs,)](
        x, out, N, scale, BLOCK_SIZE
    )
    
    return out

def replacement_func():
    return simple_multiply