import torch
import triton
import triton.language as tl

def pattern(a, b):
    """Simple pattern that just adds two tensors - this should be very common"""
    return a + b

def replacement_args(a, b):
    return (a, b)

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
def simple_triton_add(x, y):
    n_elements = x.numel()
    block_size = 1024
    grid_size = (n_elements + block_size - 1) // block_size
    
    output = torch.empty_like(x)
    
    simple_add_kernel[grid_size](x.contiguous(), y.contiguous(), output, n_elements, block_size)
    return output

def replacement_func():
    return simple_triton_add