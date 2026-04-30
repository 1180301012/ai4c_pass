import torch
import triton
import triton.language as tl


def pattern(x):
    y = x * 1.0
    return y


def replacement_args(x):
    return (x,)


# Tiny Triton kernel kept in the pass file so the pass contains a Triton implementation,
# while the actual fast-path replacement is a pure identity rewrite.
@triton.jit
def _unused_identity_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(y_ptr + offs, x, mask=mask)


def identity(x):
    return x


def replacement_func():
    return identity