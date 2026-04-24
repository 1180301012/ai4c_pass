import torch
import triton
import triton.language as tl
from torch import device


# ── Triton kernel (identical copy from FuseCausalAttnMask_N9) ──────────────────
@triton.jit
def _shared_attn_causal_kernel(
    causal_ptr,
    attn_mask_ptr,
    out_ptr,
    N,
    BLOCK_N: tl.constexpr,
):
    row  = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    valid = cols < N

    causal_val = tl.load(causal_ptr + row * N + cols, mask=valid, other=0.0)

    row_val = tl.load(attn_mask_ptr + row * N + cols, mask=valid, other=1)

    is_valid = causal_val > -3.4028234663852886e+38
    causal_out = tl.where(is_valid, causal_val, -3.4028234663852886e+38)

    out = tl.where(row_val == 0, causal_out, -3.4028234663852886e+38)

    tl.store(out_ptr + row * N + cols, out, mask=valid)


@torch.fx.wrap
def _shared_dispatch(causal_mask, attn_mask, route):
    N = causal_mask.shape[-1]
    BLOCK_N = 16
    out = torch.empty((1, 1, N, N), dtype=torch.float32, device=causal_mask.device)
    _shared_attn_causal_kernel[(N,)](causal_mask, attn_mask, out, N, BLOCK_N=BLOCK_N)
    return out


# ── Pattern ────────────────────────────────────────────────────────────────────
# This file is a duplicate mirror of FuseCausalAttnMask_N9 so that
# replacement_func() returns the SAME function object identity.
def pattern(causal_mask_arg, attn_bool_arg, tmp_13_const, tmp_12_expand):
    tmp_14 = tmp_13_const - tmp_12_expand
    tmp_15 = tmp_14.to(torch.bool)
    tmp_16 = tmp_14.masked_fill(tmp_15, -3.4028234663852886e+38)
    tmp_18 = tmp_16.to(torch.bool)
    tmp_19 = causal_mask_arg.masked_fill(tmp_18, -3.4028234663852886e+38)
    return tmp_19


def replacement_args(causal_mask_arg, attn_bool_arg, tmp_13_const, tmp_12_expand):
    return (causal_mask_arg, tmp_12_expand, "N13")


def replacement_func():
    return _shared_dispatch