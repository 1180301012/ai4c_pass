"""
Generic fused apply (backup pass v2): same logic as v1 but distinct Triton/wrapper names.
"""

import torch
import triton
import triton.language as tl
from torch import device


@triton.jit
def _apply_padding_mask_flat_v2(
    tmp9_ptr,
    tmp14_ptr,
    out_ptr,
    N_total,
    BLOCK: tl.constexpr,
):
    NEG_INF = -3.4028234663852886e+38
    offs    = tl.arange(0, BLOCK)
    mask    = offs < N_total

    causal  = tl.load(tmp9_ptr  + offs, mask=mask, other=0.0)
    pad_val = tl.load(tmp14_ptr + offs, mask=mask, other=0.0)

    out = tl.where(pad_val != 0.0, NEG_INF, causal)
    tl.store(out_ptr + offs, out.to(tl.float32), mask=mask)


@torch.fx.wrap
def apply_padding_mask_v2(tmp_9: torch.Tensor, tmp_14: torch.Tensor) -> torch.Tensor:
    NEG_INF = -3.4028234663852886e+38
    return tmp_9.masked_fill_(tmp_14.bool(), NEG_INF)


def pattern(tmp_9, tmp_14):
    tmp_15 = tmp_14.to(torch.bool)
    tmp_16 = tmp_14.masked_fill(tmp_15, -3.4028234663852886e+38)
    tmp_17 = tmp_16.to(device(type='cuda', index=0))
    tmp_18 = tmp_17.bool()
    tmp_19 = tmp_9.masked_fill(tmp_18, -3.4028234663852886e+38)
    return tmp_19


def replacement_args(tmp_9, tmp_14):
    return (tmp_9, tmp_14)


def replacement_func():
    return apply_padding_mask_v2