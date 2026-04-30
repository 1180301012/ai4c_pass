import torch
import triton
import triton.language as tl

from pass_dir.iadd_transpose_pattern_helper import pattern


# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def _specialized_inplace_broadcast_add_1x128x19_kernel(
    bias_ptr,
    x_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < 2432

    x_vals = tl.load(x_ptr + offs, mask=mask, other=0.0)
    m = offs // 19
    bias_vals = tl.load(bias_ptr + m, mask=mask, other=0.0)
    tl.store(x_ptr + offs, x_vals + bias_vals, mask=mask)


@torch.fx.wrap
def _triton_inplace_broadcast_add(in_0, in_1):
    _specialized_inplace_broadcast_add_1x128x19_kernel[(1,)](
        in_0,
        in_1,
        BLOCK_SIZE=4096,
        num_warps=4,
        num_stages=1,
    )
    return in_1


# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return _triton_inplace_broadcast_add