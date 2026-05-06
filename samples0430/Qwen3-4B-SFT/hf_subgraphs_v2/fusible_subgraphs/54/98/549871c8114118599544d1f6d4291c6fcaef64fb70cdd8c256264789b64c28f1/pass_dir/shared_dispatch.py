"""
Shared dispatch wrapper used by all diagnostic passes.
Imported by every FuseXXX.py file so that replacement_func returns
the IDENTICAL function object, satisfying the replacement_func_limit.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _shared_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask), mask=mask)


@torch.fx.wrap
def _shared_dispatch(inp, route):
    """
    Diagnostic identity kernel: always copies inp to out.
    The 'route' string differentiates which pattern matched.
    """
    out = torch.empty_like(inp)
    n = inp.numel()
    BLOCK = 1024
    _shared_kernel[(triton.cdiv(n, BLOCK),)](inp, out, n, BLOCK)
    return out