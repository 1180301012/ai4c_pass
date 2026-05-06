import torch
import triton
import triton.language as tl


@triton.jit
def _add2_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x   = tl.load(x_ptr + offs, mask=mask)
    y   = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


@torch.fx.wrap
def _diag_add(x, y):
    out = torch.empty_like(x)
    N   = x.numel()
    _add2_kernel[((N + 255) // 256,)](x, y, out, N, BLOCK=256)
    return out


# Pattern: start with the most minimal add we can possibly match
def pattern(x, y):
    return x + y


def replacement_args(x, y):
    return (x, y)


def replacement_func():
    return _diag_add