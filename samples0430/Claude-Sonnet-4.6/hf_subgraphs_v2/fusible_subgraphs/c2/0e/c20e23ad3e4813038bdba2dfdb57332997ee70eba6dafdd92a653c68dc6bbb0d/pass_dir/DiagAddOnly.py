import torch
import triton
import triton.language as tl


def pattern(x, y):
    """Simplest possible pattern: element-wise add, mirrors 'in_2 + tmp_3' in model.py"""
    return x + y


def replacement_args(x, y):
    return (x, y)


@triton.jit
def _add_kernel(
    x_ptr, y_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, x + y, mask=mask)


@torch.fx.wrap
def triton_add(x, y):
    n = x.numel()
    out = torch.empty_like(x)
    BLOCK_SIZE = 1024
    _add_kernel[(n + BLOCK_SIZE - 1) // BLOCK_SIZE,](x, y, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out


def replacement_func():
    return triton_add