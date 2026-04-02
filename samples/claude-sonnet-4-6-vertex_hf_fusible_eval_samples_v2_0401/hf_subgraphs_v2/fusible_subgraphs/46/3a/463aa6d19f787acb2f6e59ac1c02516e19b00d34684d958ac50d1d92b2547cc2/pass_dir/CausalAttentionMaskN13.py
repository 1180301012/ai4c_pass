"""
Pass: CausalAttentionMaskN13
Fuses the full causal + padding attention mask computation for N=13 into one Triton kernel.

Same strategy as CausalAttentionMaskN21 – see that file for detailed explanation.
"""
import torch
import triton
import triton.language as tl
from torch import device


@triton.jit
def causal_attn_mask_kernel_n13(
    in0_ptr,
    out_final_ptr,
    out_inter_ptr,
    out_causal_ptr,
    B,
    N,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // N
    i = pid % N

    NEG_INF = -3.4028234663852886e+38

    j = tl.arange(0, BLOCK_N)
    col_ok = j < N

    in0 = tl.load(in0_ptr + b * N + j, mask=col_ok, other=1)

    causal_masked = j > i
    causal_val = tl.where(causal_masked & col_ok, NEG_INF, 0.0)
    tl.store(out_causal_ptr + b * N * N + i * N + j, causal_val, mask=col_ok)

    pad_masked = in0 == 0
    is_masked = causal_masked | pad_masked
    inter_val = tl.where(is_masked & col_ok, NEG_INF, 0.0)
    tl.store(out_inter_ptr + b * N * N + i * N + j, inter_val, mask=col_ok)

    n_valid = tl.sum(((~is_masked) & col_ok).to(tl.int32))
    scale = tl.where(n_valid > 0, 1.0, 0.0)
    final_val = inter_val * scale
    tl.store(out_final_ptr + b * N * N + i * N + j, final_val, mask=col_ok)


@torch.fx.wrap
def causal_attn_mask_13(in_0):
    B = in_0.shape[0]
    N = in_0.shape[1]

    out_final  = torch.empty(B, 1, N, N, dtype=torch.float32, device=in_0.device)
    out_inter  = torch.empty(B, 1, N, N, dtype=torch.float32, device=in_0.device)
    out_causal = torch.empty(B, 1, N, N, dtype=torch.float32, device=in_0.device)

    causal_attn_mask_kernel_n13[(B * N,)](
        in_0,
        out_final, out_inter, out_causal,
        B, N,
        BLOCK_N=16,
    )
    return out_final, out_inter, out_causal


def pattern(in_0):
    tmp_1 = torch.arange(0, 13, device=device(type='cuda', index=0))
    tmp_2 = torch.full((13, 13), fill_value=-3.4028234663852886e+38, dtype=torch.float32, device=device(type='cuda', index=0))
    tmp_3 = torch.triu(tmp_2, diagonal=1)
    tmp_4 = torch.arange(13, device=device(type='cuda', index=0))
    tmp_5 = tmp_1.reshape(-1, 1)
    tmp_6 = tmp_4 > tmp_5
    tmp_3 *= tmp_6
    tmp_7 = tmp_3
    tmp_8 = tmp_7[(None, None, slice(None, None, None), slice(None, None, None))]
    tmp_9 = tmp_8.expand(1, 1, -1, -1)
    tmp_10 = tmp_9.clone()
    tmp_11 = tmp_10[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 13, None))]
    tmp_12 = in_0[(slice(None, None, None), None, None, slice(None, None, None))]
    tmp_13 = tmp_12.to(device(type='cuda', index=0))
    tmp_14 = tmp_11 + tmp_13
    tmp_15 = tmp_14.__eq__(0)
    tmp_16 = tmp_10[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 13, None))]
    tmp_17 = tmp_16.masked_fill(tmp_15, -3.4028234663852886e+38)
    # setitem intentionally omitted
    tmp_19 = tmp_10.__eq__(-3.4028234663852886e+38)
    tmp_20 = torch.all(tmp_19, dim=-1, keepdim=True)
    tmp_21 = ~tmp_20
    tmp_22 = tmp_10.mul(tmp_21)
    return tmp_22, tmp_17, tmp_10


def replacement_args(in_0):
    return (in_0,)


def replacement_func():
    return causal_attn_mask_13