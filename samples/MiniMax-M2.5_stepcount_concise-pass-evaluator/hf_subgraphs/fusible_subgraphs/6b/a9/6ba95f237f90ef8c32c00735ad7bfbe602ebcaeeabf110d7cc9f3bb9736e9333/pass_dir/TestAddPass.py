import torch
import triton
import triton.language as tl


def pattern(a, b):
    """Match a + b"""
    return a + b


def replacement_args(a, b):
    return (a, b)


@triton.jit
def add_kernel(a_ptr, b_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    result = a + b
    
    tl.store(output_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def triton_add(a, b):
    n = a.numel()
    BLOCK_SIZE = 1024
    num_programs = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.empty_like(a)
    
    add_kernel[(num_programs,)](
        a, b, output, n, BLOCK_SIZE
    )
    
    return output


def replacement_func():
    return triton_add