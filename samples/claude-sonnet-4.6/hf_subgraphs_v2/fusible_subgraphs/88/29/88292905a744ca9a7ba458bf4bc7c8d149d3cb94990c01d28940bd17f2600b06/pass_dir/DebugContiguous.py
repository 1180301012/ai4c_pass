"""
Debug pass: matches the single operation x.contiguous().
Used to test whether basic FX pattern matching works.
If this matches, we know the framework uses call_method("contiguous").
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _memcopy_kernel(src_ptr, dst_ptr, n, BLOCK: tl.constexpr):
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(src_ptr + offs, mask=mask)
    tl.store(dst_ptr + offs, x, mask=mask)


@torch.fx.wrap
def memcopy_contiguous(x):
    n   = x.numel()
    out = torch.empty_like(x)
    _memcopy_kernel[(triton.cdiv(n, 1024),)](x, out, n, BLOCK=1024)
    return out


def pattern(x):
    return x.contiguous()


def replacement_args(x):
    return (x,)


def replacement_func():
    return memcopy_contiguous