import torch
import triton
import triton.language as tl
from pass_dir.shared_identity import identity


def pattern(x):
    y = x.tile([1, 1, 1])
    return y


def replacement_args(x):
    return (x,)


@triton.jit
def _noop_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, x, mask=mask)


def replacement_func():
    return identity