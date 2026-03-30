import torch
import triton
import triton.language as tl
from torch import device

# ── Full-computation Triton kernel for N=9 ────────────────────────────────────
# Replaces ALL ops in the graph (full, arange, masked_fill, to, expand, sub, ...)
# Input:  in_0  [1, 9]  int64 attention mask  (0=padding, nonzero=valid)
# Output: [1, 1, 9, 9]  float32
# out[0,0,i,j] = 0.0  if j <= i AND in_0[0,j] != 0
#              = NEG_INF  otherwise
@triton.jit
def _full_causal_mask_kernel_9(
    in_ptr,   # int64  [N]  attention mask (row 0 of [1,N] tensor)
    out_ptr,  # float32 [N*N]
    BLOCK_SIZE: tl.constexpr,
):
    N: tl.constexpr = 9
    NEG_INF = -3.4028234663852886e+38

    offsets = tl.arange(0, BLOCK_SIZE)
    valid   = offsets < N * N

    row = offsets // N    # i
    col = offsets % N     # j

    causal_ok = col <= row                              # causal mask
    attn_val  = tl.load(in_ptr + col, mask=valid, other=0)
    attn_ok   = attn_val != 0                          # attention mask

    out = tl.where(causal_ok & attn_ok, 0.0, NEG_INF)
    tl.store(out_ptr + offsets, out, mask=valid)


@torch.fx.wrap
def _fused_full_causal_mask_9(in_0):
    N = 9
    out = torch.empty((1, 1, N, N), dtype=torch.float32, device=in_0.device)
    # Single CTA handles all 81 elements; BLOCK_SIZE=128 is the smallest power-of-2 >= 81
    _full_causal_mask_kernel_9[(1,)](in_0, out, BLOCK_SIZE=128)
    return out


# ── Pattern: full computation for N=9 ────────────────────────────────────────
# Key change vs original model.py:
#   tmp_14 = 1.0 - tmp_12   (scalar, avoids torch.tensor(1.0) → get_attr issue)
def pattern(in_0):
    tmp_1  = torch.full((9, 9), -3.4028234663852886e+38, device=device(type='cuda', index=0))
    tmp_2  = torch.arange(9, device=device(type='cuda', index=0))
    tmp_3  = tmp_2 + 1
    tmp_4  = tmp_3.view(9, 1)
    tmp_5  = tmp_2 < tmp_4
    tmp_6  = tmp_1.masked_fill_(tmp_5, 0)
    tmp_7  = tmp_1.to(torch.float32)
    tmp_8  = tmp_7[(None, None, slice(None, None, None), slice(None, None, None))]
    tmp_9  = tmp_8.expand(1, 1, 9, 9)
    tmp_10 = in_0[(slice(None, None, None), None, None, slice(None, None, None))]
    tmp_11 = tmp_10.expand(1, 1, 9, 9)
    tmp_12 = tmp_11.to(torch.float32)
    tmp_14 = 1.0 - tmp_12                              # scalar, not torch.tensor(1.0)
    tmp_15 = tmp_14.to(torch.bool)
    tmp_16 = tmp_14.masked_fill(tmp_15, -3.4028234663852886e+38)
    tmp_17 = tmp_16.to(device(type='cuda', index=0))
    tmp_18 = tmp_17.bool()
    tmp_19 = tmp_9.masked_fill(tmp_18, -3.4028234663852886e+38)
    return tmp_19


def replacement_args(in_0):
    return (in_0,)


def replacement_func():
    return _fused_full_causal_mask_9