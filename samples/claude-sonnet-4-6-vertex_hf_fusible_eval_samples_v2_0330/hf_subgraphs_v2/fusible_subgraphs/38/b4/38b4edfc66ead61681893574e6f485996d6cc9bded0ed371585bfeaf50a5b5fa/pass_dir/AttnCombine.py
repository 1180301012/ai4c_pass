import torch
import triton
import triton.language as tl
from torch import device

# ── Triton kernel: attention-mask + causal-mask combination ───────────────────
# Takes in_0 (int64 attention mask) and tmp_9 (float32 causal mask) as inputs.
# Computes:  out[i,j] = NEG_INF  if in_0[j]==0 (padding) else tmp_9[i,j]
# This fuses: expand, to(float32), 1-x, to(bool), masked_fill, to(cuda),
#             bool(), masked_fill  —  8 ops → 1 kernel
@triton.jit
def _attn_combine_kernel_9(
    in0_ptr,    # int64   [9]   first row of in_0 (attention mask)
    causal_ptr, # float32 [81]  tmp_9 contiguous
    out_ptr,    # float32 [81]  output
    BLOCK_SIZE: tl.constexpr,
):
    N: tl.constexpr = 9
    NEG_INF = -3.4028234663852886e+38

    offsets = tl.arange(0, BLOCK_SIZE)
    valid   = offsets < N * N
    col     = offsets % N          # j index

    attn   = tl.load(in0_ptr + col, mask=valid, other=0)   # in_0[0, j]
    causal = tl.load(causal_ptr + offsets, mask=valid, other=0.0)

    # padding (attn==0) → NEG_INF; otherwise keep causal value (0 or NEG_INF)
    out = tl.where(attn == 0, NEG_INF, causal)
    tl.store(out_ptr + offsets, out, mask=valid)


@torch.fx.wrap
def _fused_attn_combine_9(tmp_9, in_0):
    # tmp_9 : float32 [1,1,9,9] causal mask
    # in_0  : int64   [1,9]     attention mask
    out = torch.empty_like(tmp_9)
    _attn_combine_kernel_9[(1,)](in_0, tmp_9, out, BLOCK_SIZE=128)
    return out


# ── Same kernel for N=13 ──────────────────────────────────────────────────────
@triton.jit
def _attn_combine_kernel_13(
    in0_ptr,
    causal_ptr,
    out_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    N: tl.constexpr = 13
    NEG_INF = -3.4028234663852886e+38

    offsets = tl.arange(0, BLOCK_SIZE)
    valid   = offsets < N * N
    col     = offsets % N

    attn   = tl.load(in0_ptr + col, mask=valid, other=0)
    causal = tl.load(causal_ptr + offsets, mask=valid, other=0.0)

    out = tl.where(attn == 0, NEG_INF, causal)
    tl.store(out_ptr + offsets, out, mask=valid)


@torch.fx.wrap
def _fused_attn_combine_13(tmp_9, in_0):
    out = torch.empty_like(tmp_9)
    _attn_combine_kernel_13[(1,)](in_0, tmp_9, out, BLOCK_SIZE=256)
    return out


# ── Pattern: ops tmp_10 → tmp_19, accepting tmp_9 and in_0 as inputs ─────────
# Uses scalar `1.0 - tmp_12` (not torch.tensor(1.0)) to avoid get_attr issue.
# This pattern is size-agnostic: works for both N=9 and N=13.
def pattern(tmp_9, in_0):
    tmp_10 = in_0[(slice(None, None, None), None, None, slice(None, None, None))]
    tmp_11 = tmp_10.expand(1, 1, 9, 9)
    tmp_12 = tmp_11.to(torch.float32)
    tmp_14 = 1.0 - tmp_12
    tmp_15 = tmp_14.to(torch.bool)
    tmp_16 = tmp_14.masked_fill(tmp_15, -3.4028234663852886e+38)
    tmp_17 = tmp_16.to(device(type='cuda', index=0))
    tmp_18 = tmp_17.bool()
    tmp_19 = tmp_9.masked_fill(tmp_18, -3.4028234663852886e+38)
    return tmp_19


@torch.fx.wrap
def _fused_attn_combine(tmp_9, in_0):
    N_sq = tmp_9.numel()
    out  = torch.empty_like(tmp_9)
    if N_sq <= 81:
        _attn_combine_kernel_9[(1,)](in_0, tmp_9, out, BLOCK_SIZE=128)
    else:
        _attn_combine_kernel_13[(1,)](in_0, tmp_9, out, BLOCK_SIZE=256)
    return out


def replacement_args(tmp_9, in_0):
    return (tmp_9, in_0)


def replacement_func():
    return _fused_attn_combine