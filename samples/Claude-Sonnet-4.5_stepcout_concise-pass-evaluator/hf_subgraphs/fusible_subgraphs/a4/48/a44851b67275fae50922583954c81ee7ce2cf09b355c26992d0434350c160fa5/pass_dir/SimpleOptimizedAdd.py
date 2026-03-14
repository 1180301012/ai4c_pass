import torch
import triton
import triton.language as tl


def pattern(in_2, in_3):
    """Pattern: Just add - will be faster with optimized kernel"""
    result = in_2 + in_3
    return result


def replacement_args(in_2, in_3):
    return (in_2, in_3)


@triton.jit
def optimized_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Very simple optimized add kernel"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def optimized_add_simple(x, y):
    """Simple optimized add"""
    if not x.is_cuda:
        return x + y
    
    n_elements = x.numel()
    out = torch.empty_like(x)
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    optimized_add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return out


def replacement_func():
    return optimized_add_simple