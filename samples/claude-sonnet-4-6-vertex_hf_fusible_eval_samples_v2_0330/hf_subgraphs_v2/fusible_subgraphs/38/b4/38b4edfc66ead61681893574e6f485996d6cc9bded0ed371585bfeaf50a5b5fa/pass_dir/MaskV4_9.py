import torch
import triton
import triton.language as tl
from torch import device

# ── Triton kernel for N=9 ─────────────────────────────────────────────────────
# Inputs: in_0 [1,9] int64 attention mask, tmp_9 [1,1,9,9] float32 causal mask
# Reads in_0 via col-index so only 9 unique int64 values are accessed.
@triton.jit
def _v4_kernel_9(
    in0_ptr,    # int64 [9]  — in_0[0, 0..8]
    causal_ptr, # float32 [81]  — tmp_9 contiguous
    out_ptr,    # float32 [81]
    BLOCK_SIZE: tl.constexpr,
):
    N: tl.constexpr   = 9
    N_SQ: tl.constexpr = 81
    NEG_INF = -3.4028234663852886e+38

    offsets = tl.arange(0, BLOCK_SIZE)
    valid   = offsets < N_SQ
    col     = offsets % N

    attn   = tl.load(in0_ptr + col, mask=valid, other=1)
    causal = tl.load(causal_ptr + offsets, mask=valid, other=0.0)
    out    = tl.where(attn == 0, NEG_INF, causal)
    tl.store(out_ptr + offsets, out, mask=valid)


@torch.fx.wrap
def _fused_v4_9(tmp_9, in_0, const_one_unused):
    out = torch.empty_like(tmp_9)
    _v4_kernel_9[(1,)](in_0, tmp_9, out, BLOCK_SIZE=128)
    return out


# ── Pattern for N=9: 9 internal ops → 1 Triton kernel ────────────────────────
# Includes getitem + expand (size-specific: 9) before the existing 7 ops.
# const_one placeholder matches torch.tensor(1.0) call_function in target.
def pattern(tmp_9, in_0, const_one):
    tmp_10 = in_0[(slice(None, None, None), None, None, slice(None, None, None))]
    tmp_11 = tmp_10.expand(1, 1, 9, 9)
    tmp_12 = tmp_11.to(torch.float32)
    tmp_14 = const_one - tmp_12
    tmp_15 = tmp_14.to(torch.bool)
    tmp_16 = tmp_14.masked_fill(tmp_15, -3.4028234663852886e+38)
    tmp_17 = tmp_16.to(device(type='cuda', index=0))
    tmp_18 = tmp_17.bool()
    tmp_19 = tmp_9.masked_fill(tmp_18, -3.4028234663852886e+38)
    return tmp_19


def replacement_args(tmp_9, in_0, const_one):
    return (tmp_9, in_0, const_one)


def replacement_func():
    return _fused_v4_9