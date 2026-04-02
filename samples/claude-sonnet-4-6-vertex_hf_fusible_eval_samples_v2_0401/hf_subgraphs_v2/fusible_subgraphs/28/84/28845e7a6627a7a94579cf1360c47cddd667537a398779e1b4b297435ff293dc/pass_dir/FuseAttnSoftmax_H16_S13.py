"""
Fused pass: add + max_clamp + view(16,13,13) + softmax(dim=-1) + dropout(training=False)
Covers: float16 xglm-564M
"""

import torch
from torch import device
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern (must mirror model.py exactly)
# ---------------------------------------------------------------------------
def pattern(in_0, in_1):
    tmp_0 = in_1 + in_0
    tmp_1 = torch.tensor(-3.4028234663852886e+38, device=device(type='cuda', index=0))
    tmp_2 = torch.max(tmp_0, tmp_1)
    tmp_3 = tmp_2.view(16, 13, 13)
    tmp_4 = torch.nn.functional.softmax(tmp_3, dim=-1)
    tmp_5 = torch.nn.functional.dropout(tmp_4, p=0.1, training=False)
    return (tmp_5,)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------
@triton.jit
def _fused_attn_softmax_kernel(
    in0_ptr,   # [1, 1, S, S]
    in1_ptr,   # [1, H, S, S]
    out_ptr,   # [H, S, S]
    H, S,
    in0_rs, in0_cs,          # in0 row-stride, col-stride
    in1_hs, in1_rs, in1_cs,  # in1 head/row/col strides
    BLOCK_S: tl.constexpr,
    DTYPE: tl.constexpr,     # 1=fp16, 2=bf16, 3=fp32
):
    row_id = tl.program_id(0)
    h = row_id // S
    r = row_id % S

    cols = tl.arange(0, BLOCK_S)
    mask_c = cols < S

    # Load attention mask (broadcast over heads: always [0, 0, r, c])
    in0_offs = r * in0_rs + cols * in0_cs
    x0 = tl.load(in0_ptr + in0_offs, mask=mask_c, other=float('-inf')).to(tl.float32)

    # Load attention scores [0, h, r, c]
    in1_offs = h * in1_hs + r * in1_rs + cols * in1_cs
    x1 = tl.load(in1_ptr + in1_offs, mask=mask_c, other=0.0).to(tl.float32)

    # Fused: add + clamp(-3.4028e38)
    x = x0 + x1
    x = tl.maximum(x, -3.4028234663852886e+38)

    # Softmax (numerically stable)
    x_max = tl.max(x, axis=0)
    x = x - x_max
    x_exp = tl.exp(x)
    x_sum = tl.sum(x_exp, axis=0)
    x_out = x_exp / x_sum

    # Store with dtype conversion
    out_offs = (h * S + r) * S + cols
    if DTYPE == 1:
        tl.store(out_ptr + out_offs, x_out.to(tl.float16), mask=mask_c)
    elif DTYPE == 2:
        tl.store(out_ptr + out_offs, x_out.to(tl.bfloat16), mask=mask_c)
    else:
        tl.store(out_ptr + out_offs, x_out, mask=mask_c)


_DTYPE_MAP = {torch.float16: 1, torch.bfloat16: 2, torch.float32: 3}

# ---------------------------------------------------------------------------
# Replacement wrapper
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_attn_softmax_H16_S13(in_0, in_1):
    H, S = 16, 13
    BLOCK_S = 16  # next power-of-2 >= S=13

    dtype = in_1.dtype
    DTYPE = _DTYPE_MAP.get(dtype, 3)

    out = torch.empty((H, S, S), dtype=dtype, device=in_1.device)

    grid = (H * S,)
    _fused_attn_softmax_kernel[grid](
        in_0, in_1, out,
        H, S,
        in_0.stride(2), in_0.stride(3),
        in_1.stride(1), in_1.stride(2), in_1.stride(3),
        BLOCK_S=BLOCK_S,
        DTYPE=DTYPE,
    )
    return (out,)


def replacement_func():
    return fused_attn_softmax_H16_S13