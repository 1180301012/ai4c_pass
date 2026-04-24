"""
Diagnostic pass: match just add (in_1 + in_0) to confirm pattern matching works.
If this matches, the issue is with softmax; if it doesn't, there's a framework issue.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, x + y, mask=mask)


@torch.fx.wrap
def triton_add(in_1, in_0):
    out = torch.empty_like(in_1)
    n = in_1.numel()
    BLOCK = 1024
    grid = ((n + BLOCK - 1) // BLOCK,)
    _add_kernel[grid](in_1, in_0, out, n, BLOCK)
    return out


def pattern(in_1, in_0):
    return in_1 + in_0


def replacement_args(in_1, in_0):
    return (in_1, in_0)


def replacement_func():
    return triton_add