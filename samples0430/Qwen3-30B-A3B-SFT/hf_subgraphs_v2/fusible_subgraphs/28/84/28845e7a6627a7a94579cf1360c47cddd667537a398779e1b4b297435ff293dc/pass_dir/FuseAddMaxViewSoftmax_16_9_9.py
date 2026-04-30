import torch
import triton
import triton.language as tl
from torch import device


@triton.jit
def _triton_softmax_kernel_16_9(
    x_ptr,
    out_ptr,
    S,
    BLOCK_S: tl.constexpr,
):
    """Softmax for last dimension of 3D tensor [B, S, S]."""
    b = tl.program_id(0)
    r = tl.program_id(1)

    cols = tl.arange(0, BLOCK_S)
    mask = cols < S

    x = tl.load(x_ptr + b * S * S + r * S + cols, mask=mask, other=0.0).to(tl.float32)

    x_max = tl.max(x, axis=0)
    x_exp = tl.exp(x - x_max)
    x_exp = tl.where(mask, x_exp, 0.0)
    x_sum = tl.sum(x_exp, axis=0)
    out_vals = x_exp / x_sum

    tl.store(out_ptr + b * S * S + r * S + cols, out_vals, mask=mask)


@torch.fx.wrap
def _triton_softmax(in_0):
    B = in_0.shape[0]
    S = in_0.shape[1]
    out = torch.empty(B, S, S, dtype=in_0.dtype, device=in_0.device)
    _triton_softmax_kernel_16_9[(B, S)](
        in_0, out,
        S,
        BLOCK_S=16,
    )
    return out


def pattern(in_0):
    tmp_4 = torch.nn.functional.softmax(in_0, dim=-1)
    tmp_5 = torch.nn.functional.dropout(tmp_4, p=0.1, training=False)
    return tmp_5


def replacement_args(in_0):
    return (in_0,)


def replacement_func():
    return _triton_softmax