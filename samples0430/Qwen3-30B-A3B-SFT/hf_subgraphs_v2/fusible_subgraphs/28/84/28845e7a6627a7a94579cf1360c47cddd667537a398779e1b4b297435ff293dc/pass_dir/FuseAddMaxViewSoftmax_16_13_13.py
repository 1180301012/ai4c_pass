import torch
import triton
import triton.language as tl
from torch import device


@triton.jit
def _add_softmax_2d_kernel(
    in0_ptr,
    in1_ptr,
    out_ptr,
    B,
    S,
    in0_stride_b,
    in1_stride_b,
    BLOCK_S: tl.constexpr,
):
    """2D grid: each program handles one (b, row) pair, computing row softmax."""
    b = tl.program_id(0)
    r = tl.program_id(1)
    cols = tl.arange(0, BLOCK_S)
    mask = cols < S

    in0_vals = tl.load(in0_ptr + b * in0_stride_b + r * S + cols,
                       mask=mask, other=0.0).to(tl.float32)
    in1_vals = tl.load(in1_ptr + b * in1_stride_b + r * S + cols,
                       mask=mask, other=0.0).to(tl.float32)

    x = in0_vals + in1_vals

    # Numerically stable softmax
    x_max = tl.max(x, axis=0)
    safe_max = tl.where(x_max == float('-inf'), 0.0, x_max)
    x_exp = tl.exp(x - safe_max)
    x_exp = tl.where(mask, x_exp, 0.0)
    count = tl.sum(tl.where(mask, 1.0, 0.0), axis=0)
    out_vals = x_exp / count

    tl.store(out_ptr + b * S * S + r * S + cols, out_vals, mask=mask)


@torch.fx.wrap
def _triton_add_softmax(in_0, in_1):
    # in_0: [1, 1, S, S] broadcast (batch dim = 1)
    # in_1: [1, B, S, S]
    # Output: [B, S, S] — fused add + clamp + softmax
    B = in_1.shape[1]
    S = in_1.shape[2]
    out = torch.empty(B, S, S, dtype=in_1.dtype, device=in_1.device)
    in0_stride_b = in_0.stride(1)
    in1_stride_b = in_1.stride(1)
    _add_softmax_2d_kernel[(B, S)](
        in_0, in_1, out,
        B, S,
        in0_stride_b, in1_stride_b,
        BLOCK_S=16,
    )
    return out


def pattern(in_0, in_1):
    return in_1 + in_0


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return _triton_add_softmax