import torch
import triton
import triton.language as tl

def pattern(x, y):
    """Simple pattern for subtraction operation"""
    return x - y

def replacement_args(x, y):
    return (x, y)

@triton.jit
def sub_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Compute
    out = x - y
    
    # Store
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def triton_sub(x, y):
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    sub_kernel[(num_programs,)](
        x, y, out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return triton_sub