import torch
import triton
import triton.language as tl
from torch import device

# ---------------------------------------------------------------------------
# Shared Triton kernel (same as AttentionMask_N21.py)
# ---------------------------------------------------------------------------
_NEG_INF = -3.4028234663852886e+38


@triton.jit
def _attn_causal_mask_kernel_n10(
    in0_ptr,
    out_ptr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    col_mask = cols < N

    attn = tl.load(in0_ptr + cols, mask=col_mask, other=1)

    valid_in_window = (cols <= row) & (attn != 0)
    n_valid = tl.sum(valid_in_window.to(tl.int32), axis=0)

    is_causal = cols > row
    is_pad    = attn == 0
    is_masked = is_causal | is_pad

    base_val = tl.where(is_masked,
                        tl.full([BLOCK_N], _NEG_INF, dtype=tl.float32),
                        tl.zeros([BLOCK_N], dtype=tl.float32))

    if n_valid > 0:
        out_val = base_val
    else:
        out_val = tl.zeros([BLOCK_N], dtype=tl.float32)

    out_offs = row * N + cols
    tl.store(out_ptr + out_offs, out_val, mask=col_mask)


# ---------------------------------------------------------------------------
# Shared dispatch wrapper — identical structure to AttentionMask_N21.py
# ---------------------------------------------------------------------------
@torch.fx.wrap
def _run_attn_mask(in_0, route):
    if route == "N21":
        N = 21
        BLOCK_N = 32
    elif route == "N10":
        N = 10
        BLOCK_N = 16
    elif route == "N13":
        N = 13
        BLOCK_N = 16
    else:
        N = in_0.shape[1]
        BLOCK_N = 32

    out = torch.empty((1, 1, N, N), dtype=torch.float32, device=in_0.device)
    _attn_causal_mask_kernel_n10[(N,)](
        in_0,
        out,
        N=N,
        BLOCK_N=BLOCK_N,
    )
    return (out,)


# ---------------------------------------------------------------------------
# Pattern for N=10 (float16 graph)
# Key fixes:
#  1. tmp_7 = tmp_3 * tmp_6  (non-in-place; Dynamo decomposes *= to *)
#  2. Include setitem (side-effecting, mutates clone in-place)
#  3. Use tmp_10 (clone) — NOT tmp_17 — for the final eq/mul ops
# ---------------------------------------------------------------------------
def pattern(in_0):
    tmp_1 = torch.arange(0, 10, device=device(type='cuda', index=0))
    tmp_2 = torch.full((10, 10), fill_value=-3.4028234663852886e+38, dtype=torch.float32, device=device(type='cuda', index=0))
    tmp_3 = torch.triu(tmp_2, diagonal=1)
    tmp_4 = torch.arange(10, device=device(type='cuda', index=0))
    tmp_5 = tmp_1.reshape(-1, 1)
    tmp_6 = tmp_4 > tmp_5
    tmp_7 = tmp_3 * tmp_6                    # non-in-place mul
    tmp_8 = tmp_7[(None, None, slice(None, None, None), slice(None, None, None))]
    tmp_9 = tmp_8.expand(1, 1, -1, -1)
    tmp_10 = tmp_9.clone()
    tmp_11 = tmp_10[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 10, None))]
    tmp_12 = in_0[(slice(None, None, None), None, None, slice(None, None, None))]
    tmp_13 = tmp_12.to(device(type='cuda', index=0))
    tmp_14 = tmp_11 + tmp_13
    tmp_15 = tmp_14.__eq__(0)
    tmp_16 = tmp_10[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 10, None))]
    tmp_17 = tmp_16.masked_fill(tmp_15, -3.4028234663852886e+38)
    tmp_10[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 10, None))] = tmp_17
    tmp_19 = tmp_10.__eq__(-3.4028234663852886e+38)   # uses clone (tmp_10)
    tmp_20 = torch.all(tmp_19, dim=-1, keepdim=True)
    tmp_21 = ~tmp_20
    tmp_22 = tmp_10.mul(tmp_21)                        # uses clone (tmp_10)
    return (tmp_22,)


def replacement_args(in_0):
    return (in_0, "N10")


def replacement_func():
    return _run_attn_mask