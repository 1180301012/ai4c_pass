import torch
import triton
import triton.language as tl

_INV_282 = 0.35355339059327373   # 1.0 / 2.8284271247461903


@triton.jit
def _kern282(
    input_ptr,
    output_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """out[i] = in[i] * (1.0 / 2.8284271247461903)"""
    INV = 0.35355339059327373
    pid  = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(input_ptr + offs, mask=mask, other=0.0)
    tl.store(output_ptr + offs, x * INV, mask=mask)


@torch.fx.wrap
def _fused_div_transpose_282_wrapper(in_0):
    # Use PyTorch's native mul for minimum Python dispatch overhead.
    return in_0.mul(_INV_282).transpose(-1, -2)


# ---------------------------------------------------------------------------
# Pass interface
# ---------------------------------------------------------------------------

def pattern(in_0):
    tmp_0 = in_0 / 2.8284271247461903
    tmp_1 = tmp_0.transpose(-1, -2)
    return tmp_1


def replacement_args(in_0):
    return (in_0,)


def replacement_func():
    return _fused_div_transpose_282_wrapper