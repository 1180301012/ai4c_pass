import torch
import triton
import triton.language as tl

def pattern(x):
    """Simple test pattern"""
    return x * 2.0

def replacement_args(x):
    return (x,)

@triton.jit
def simple_test_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Simple test kernel"""
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = x * 2.0
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def simple_test(x):
    """Simple test function"""
    out = torch.empty_like(x)
    total_elements = x.numel()
    BLOCK_SIZE = 1024
    grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    simple_test_kernel[(grid_size,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=total_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return simple_test