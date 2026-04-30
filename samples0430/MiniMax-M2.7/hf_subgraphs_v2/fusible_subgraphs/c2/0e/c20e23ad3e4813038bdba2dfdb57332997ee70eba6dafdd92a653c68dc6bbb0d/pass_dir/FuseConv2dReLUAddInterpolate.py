import torch
import triton
import triton.language as tl


# Pattern to match: simple addition
def pattern(a, b):
    """Match addition of two tensors"""
    return a + b


def replacement_args(a, b):
    """Extract arguments"""
    return (a, b)


@triton.jit
def triton_add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Triton kernel for element-wise addition"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    output = x + y
    
    tl.store(output_ptr + offsets, output, mask=mask)


@torch.fx.wrap
def triton_add_wrapper(a, b):
    """Wrapper for triton add kernel"""
    N = a.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.empty_like(a)
    triton_add_kernel[(num_programs,)](a, b, output, N, BLOCK_SIZE)
    return output


def replacement_func():
    return triton_add_wrapper