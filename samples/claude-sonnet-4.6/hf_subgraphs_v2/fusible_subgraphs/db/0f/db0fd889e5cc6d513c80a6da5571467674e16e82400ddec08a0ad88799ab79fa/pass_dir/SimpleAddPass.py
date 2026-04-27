"""
Diagnostic pass: match simple a + b addition.
This tests whether the FX pattern matching framework is working at all.
Matches: tmp_23 = tmp_12 + tmp_22  (lines 24 in model.py)
"""

import torch
import triton
import triton.language as tl


def pattern(a, b):
    return a + b


def replacement_args(a, b):
    return (a, b)


@triton.jit
def _triton_add_kernel(a_ptr, b_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, a + b, mask=mask)


@torch.fx.wrap
def triton_add(a, b):
    n = a.numel()
    out = torch.empty_like(a)
    BLOCK = 1024
    grid = ((n + BLOCK - 1) // BLOCK,)
    _triton_add_kernel[grid](a, b, out, n, BLOCK=BLOCK)
    return out


def replacement_func():
    return triton_add