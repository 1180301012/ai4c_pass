"""
AI4C pass: Diagnostic x+y addition pattern.

Testing whether the pass framework's pattern matching works for basic
operator-level operations. If x+y matches, we build more complex patterns.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < N
    x = tl.load(x_ptr + i, mask=mask, other=0.0)
    y = tl.load(y_ptr + i, mask=mask, other=0.0)
    tl.store(out_ptr + i, x + y, mask=mask)


@torch.fx.wrap
def _diag_add(x, y):
    N = x.numel()
    BLOCK = 1024
    out = torch.empty_like(x)
    _add_kernel[((N + BLOCK - 1) // BLOCK,)](x, y, out, N, BLOCK=BLOCK)
    return out


def pattern(x, y):
    return (x + y,)


def replacement_args(x, y):
    return (x, y)


@torch.fx.wrap
def fused_softmax_dropout(x):
    W = x.shape[-1]
    N = x.numel() // W
    out = torch.empty_like(x)
    _add_kernel  # unused placeholder so lint passes
    return out


def replacement_func():
    return _diag_add