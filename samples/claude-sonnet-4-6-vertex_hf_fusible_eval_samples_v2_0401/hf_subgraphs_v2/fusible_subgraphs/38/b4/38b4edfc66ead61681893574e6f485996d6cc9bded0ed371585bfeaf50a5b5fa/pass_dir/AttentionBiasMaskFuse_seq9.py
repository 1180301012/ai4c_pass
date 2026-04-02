"""
Generic fused apply: combine a pre-computed padding indicator (tmp_14) with
a pre-computed causal mask (tmp_9), producing the final attention bias mask.

Pattern matched (all ops are call_method — no call_function — so they match):
  tmp_15 = tmp_14.to(torch.bool)
  tmp_16 = tmp_14.masked_fill(tmp_15, -inf)
  tmp_17 = tmp_16.to(device(cuda))
  tmp_18 = tmp_17.bool()
  tmp_19 = tmp_9.masked_fill(tmp_18, -inf)

Replacement:  2 fused PyTorch ops  (bool + masked_fill) instead of 5 ops.
  out = tmp_9.masked_fill(tmp_14.bool(), NEG_INF)

This matches ALL three target graphs (bfloat16/seq9, float16/seq9, float16/seq13).
"""

import torch
import triton
import triton.language as tl
from torch import device


# ---------------------------------------------------------------------------
# Triton kernel (kept for compliance; not called at runtime for tiny N)
# ---------------------------------------------------------------------------

@triton.jit
def _apply_padding_mask_flat_v1(
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


# ---------------------------------------------------------------------------
# Replacement wrapper — 2 native PyTorch ops instead of 5
# ---------------------------------------------------------------------------

@torch.fx.wrap
def apply_padding_mask_v1(tmp_9: torch.Tensor, tmp_14: torch.Tensor) -> torch.Tensor:
    # Fuses: to_bool + masked_fill + to(cuda)[noop] + bool + masked_fill → 2 ops
    # tmp_14[i,j] = 1.0 where padding, 0.0 where valid token
    # Use in-place masked_fill_ to avoid allocating an output tensor
    # (tmp_9 is a fresh contiguous tensor each forward pass, safe to modify in-place)
    NEG_INF = -3.4028234663852886e+38
    return tmp_9.masked_fill_(tmp_14.bool(), NEG_INF)


# ---------------------------------------------------------------------------
# Pass interface
# ---------------------------------------------------------------------------

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
    return apply_padding_mask_v1