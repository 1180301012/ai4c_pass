import torch
import triton
import triton.language as tl

def pattern(x, y):
    return x + y

def replacement_args(x, y):
    return (x, y)

@triton.jit
def simple_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    result = x + y
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def simple_add(x, y):
    n_elements = x.numel()
    block_size = 1024
    num_programs = (n_elements + block_size - 1) // block_size
    
    out = torch.empty_like(x)
    simple_add_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=block_size
    )
    return out

def replacement_func():
    return simple_add