import torch
import triton
import triton.language as tl


def pattern(a, b):
    return a + b


def replacement_args(a, b):
    return (a, b)


@triton.jit
def add_kernel(a_ptr, b_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    out = a + b
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def triton_add(a, b):
    out = torch.empty_like(a)
    n = a.numel()
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    add_kernel[grid](a, b, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out


def replacement_func():
    return triton_add