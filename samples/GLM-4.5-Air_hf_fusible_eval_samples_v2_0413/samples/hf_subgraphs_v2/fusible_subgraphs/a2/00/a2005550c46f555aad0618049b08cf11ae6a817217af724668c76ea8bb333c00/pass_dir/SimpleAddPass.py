import torch
import triton
import triton.language as tl

def pattern(x, y):
    return x + y

def replacement_args(x, y):
    return (x, y)

@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    output = x + y
    
    tl.store(output_ptr + offsets, output, mask=mask)

@torch.fx.wrap
def simple_add_kernel(x, y):
    n_elements = x.numel()
    block_size = 1024
    num_programs = (n_elements + block_size - 1) // block_size
    
    output = torch.empty_like(x)
    
    add_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=block_size
    )
    
    return output

def replacement_func():
    return simple_add_kernel