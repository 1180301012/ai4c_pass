import torch
from torch import device
import triton
import triton.language as tl


@triton.jit
def causal_attn_mask_kernel_N13(
    in_0_ptr,   # pointer to int64 [1, N] attention mask
    out_ptr,    # pointer to float32 [1, 1, N, N] output
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    # Load attention mask in_0[0, j] for j = 0..BLOCK_N-1 (int64)
    attn = tl.load(in_0_ptr + cols, mask=mask, other=1)
    attn_f32 = attn.to(tl.float32)

    NEG_FLTMAX = -3.4028234663852886e+38

    # Causal bias: NEG_FLTMAX if col > row, else 0.0
    causal = tl.where(cols > row, NEG_FLTMAX, 0.0).to(tl.float32)

    # combined = causal + attn_f32
    combined = causal + attn_f32

    # Masked fill: set to NEG_FLTMAX where combined == 0
    val = tl.where(combined == 0.0, NEG_FLTMAX, causal).to(tl.float32)

    # Row all-inf check: count positions (within valid cols) where val == 0.0
    row_has_valid = tl.sum((mask & (val == 0.0)).to(tl.int32), axis=0)
    row_all_inf = row_has_valid == 0

    # Zero out all-inf rows
    final = tl.where(row_all_inf, tl.float32(0.0), val)

    # Store to out[0, 0, row, j] = out_ptr + row * N + j
    tl.store(out_ptr + row * N + cols, final, mask=mask)


@torch.fx.wrap
def causal_attn_mask_N13(in_0):
    N = 13
    BLOCK_N = 16
    out = torch.empty((1, 1, N, N), dtype=torch.float32, device=in_0.device)
    causal_attn_mask_kernel_N13[(N,)](
        in_0_ptr=in_0,
        out_ptr=out,
        N=N,
        BLOCK_N=BLOCK_N,
    )
    return (out,)


def pattern(in_0):
    tmp_0 = in_0
    tmp_1 = torch.arange(0, 13, device=device(type='cuda', index=0))
    tmp_2 = torch.full((13, 13), fill_value=-3.4028234663852886e+38, dtype=torch.float32, device=device(type='cuda', index=0))
    tmp_3 = torch.triu(tmp_2, diagonal=1)
    tmp_4 = torch.arange(13, device=device(type='cuda', index=0))
    tmp_5 = tmp_1.reshape(-1, 1)
    tmp_6 = tmp_4 > tmp_5
    tmp_3 *= tmp_6
    tmp_7 = tmp_3
    tmp_8 = tmp_7[None, None, slice(None, None, None), slice(None, None, None)]
    tmp_9 = tmp_8.expand(1, 1, -1, -1)
    tmp_10 = tmp_9.clone()
    tmp_11 = tmp_10[slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 13, None)]
    tmp_12 = tmp_0[slice(None, None, None), None, None, slice(None, None, None)]
    tmp_13 = tmp_12.to(device(type='cuda', index=0))
    tmp_14 = tmp_11 + tmp_13
    tmp_15 = tmp_14.__eq__(0)
    tmp_16 = tmp_10[slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 13, None)]
    tmp_17 = tmp_16.masked_fill(tmp_15, -3.4028234663852886e+38)
    # Use tmp_17 directly (equivalent to tmp_10 after setitem since slice covers all cols)
    tmp_19 = tmp_17.__eq__(-3.4028234663852886e+38)
    tmp_20 = torch.all(tmp_19, dim=-1, keepdim=True)
    tmp_21 = ~tmp_20
    tmp_22 = tmp_17.mul(tmp_21)
    return (tmp_22,)


def replacement_args(in_0):
    return (in_0,)


def replacement_func():
    return causal_attn_mask_N13