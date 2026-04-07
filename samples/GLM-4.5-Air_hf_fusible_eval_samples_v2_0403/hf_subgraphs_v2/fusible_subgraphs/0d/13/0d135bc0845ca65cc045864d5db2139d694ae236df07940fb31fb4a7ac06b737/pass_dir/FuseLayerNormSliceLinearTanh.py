import torch
import triton
import triton.language as tl

def pattern(in_5, in_6):
    tmp_5 = in_6 + in_5
    return tmp_5

def replacement_args(in_5, in_6):
    return (in_5, in_6)

@triton.jit
def simple_add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    output = x + y
    
    tl.store(output_ptr + offsets, output, mask=mask)

@torch.fx.wrap
def triton_add_op(x, y):
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.empty_like(x)
    
    grid = (num_programs,)
    simple_add_kernel[grid](
        x, y, output, n_elements, BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return triton_add_op