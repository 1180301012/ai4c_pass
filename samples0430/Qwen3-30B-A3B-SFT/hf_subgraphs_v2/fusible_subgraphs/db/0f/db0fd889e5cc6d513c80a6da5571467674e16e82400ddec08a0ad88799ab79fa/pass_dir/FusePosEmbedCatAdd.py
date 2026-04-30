import torch
import triton
import triton.language as tl

from pass_dir.shared_dispatch import dispatch_kernel


# Minimal diagnostic: just fuse x + y to see if add matching works at all
@triton.jit
def _triton_add_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


@torch.fx.wrap
def _triton_add(x, y):
    N = x.numel()
    out = torch.empty_like(x)
    BLOCK = 1024
    grid = ((N + BLOCK - 1) // BLOCK,)
    _triton_add_kernel[grid](x, y, out, N, BLOCK)
    return out


def pattern(x, y):
    return x + y


def replacement_args(x, y):
    return (x, y)


def replacement_func():
    return _triton_add