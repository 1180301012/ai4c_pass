import torch
import triton
import triton.language as tl
from torch import device

# ── Full-computation Triton kernel for N=13 ───────────────────────────────────
@triton.jit
def _full_causal_mask_kernel_13(
    in_ptr,   # int64  [N]  attention mask (row 0 of [1,N] tensor)
    out_ptr,  # float32 [N*N]
    BLOCK_SIZE: tl.constexpr,
):
    N: tl.constexpr = 13
    NEG_INF = -3.4028234663852886e+38

    offsets = tl.arange(0, BLOCK_SIZE)
    valid   = offsets < N * N

    row = offsets // N
    col = offsets % N

    causal_ok = col <= row
    attn_val  = tl.load(in_ptr + col, mask=valid, other=0)
    attn_ok   = attn_val != 0

    out = tl.where(causal_ok & attn_ok, 0.0, NEG_INF)
    tl.store(out_ptr + offsets, out, mask=valid)


@torch.fx.wrap
def _fused_full_causal_mask_13(in_0):
    N = 13
    out = torch.empty((1, 1, N, N), dtype=torch.float32, device=in_0.device)
    # Single CTA handles all 169 elements; BLOCK_SIZE=256 >= 169
    _full_causal_mask_kernel_13[(1,)](in_0, out, BLOCK_SIZE=256)
    return out


# ── Pattern: full computation for N=13 ───────────────────────────────────────
def pattern(in_0):
    tmp_1  = torch.full((13, 13), -3.4028234663852886e+38, device=device(type='cuda', index=0))
    tmp_2  = torch.arange(13, device=device(type='cuda', index=0))
    tmp_3  = tmp_2 + 1
    tmp_4  = tmp_3.view(13, 1)
    tmp_5  = tmp_2 < tmp_4
    tmp_6  = tmp_1.masked_fill_(tmp_5, 0)
    tmp_7  = tmp_1.to(torch.float32)
    tmp_8  = tmp_7[(None, None, slice(None, None, None), slice(None, None, None))]
    tmp_9  = tmp_8.expand(1, 1, 13, 13)
    tmp_10 = in_0[(slice(None, None, None), None, None, slice(None, None, None))]
    tmp_11 = tmp_10.expand(1, 1, 13, 13)
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
    return _fused_full_causal_mask_13