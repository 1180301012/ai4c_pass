"""
FuseRelAttnH8.py  (W=8, N=64)
Matches the full relative-position attention tail for the 64×64 attention case.
"""

import torch
import triton
import triton.language as tl


def pattern(tmp7, in2, in0, v):
    tmp_8  = tmp7.expand(-1, -1, 8, -1, -1)
    tmp_9  = tmp_8.permute((0, 3, 1, 4, 2))
    tmp_10 = tmp_9 + in2
    tmp_11 = tmp_10.reshape(4, 64, 64)
    tmp_12 = in0 + tmp_11
    tmp_13 = tmp_12.softmax(dim=-1)
    matmul_1 = tmp_13 @ v
    tmp_15 = matmul_1.transpose(-1, -2)
    return tmp_15


def replacement_args(tmp7, in2, in0, v):
    return (tmp7, in2, in0, v)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=2),
    ],
    key=['B'],
)
@triton.jit
def _rel_pos_softmax_H8_kernel(
    TMP7_ptr, IN2_ptr, IN0_ptr, OUT_ptr,
    B,
    s7b, s7h, s7p, s7k,
    s2b, s2r, s2c,
    s0b, s0i,
    sob, soi,
    W: tl.constexpr = 8,
    N: tl.constexpr = 64,
):
    pid = tl.program_id(0)
    b   = pid // N
    i   = pid %  N
    q_row = i // W
    q_col = i %  W
    offs_kr = tl.arange(0, W)
    offs_kc = tl.arange(0, W)
    offs_j  = offs_kr[:, None] * W + offs_kc[None, :]

    in0_row   = tl.load(IN0_ptr  + b*s0b + i*s0i  + offs_j                       ).to(tl.float32)
    tmp7_vals = tl.load(TMP7_ptr + b*s7b + q_col*s7h + q_row*s7p + offs_kr*s7k   ).to(tl.float32)
    in2_row   = tl.load(IN2_ptr  + b*s2b + q_row*s2r + q_col*s2c + offs_j        ).to(tl.float32)

    att     = in0_row + tmp7_vals[:, None] + in2_row
    row_max = tl.max(att, axis=1)
    att_max = tl.max(row_max, axis=0)
    exp_att = tl.exp(att - att_max)
    exp_sum = tl.sum(tl.sum(exp_att, axis=1), axis=0)
    soft    = (exp_att / exp_sum).to(TMP7_ptr.dtype.element_ty)
    tl.store(OUT_ptr + b*sob + i*soi + offs_j, soft)


@torch.fx.wrap
def fused_rel_attn_H8(tmp7, in2, in0, v):
    tmp7 = tmp7.contiguous()
    in2  = in2.contiguous()
    in0  = in0.contiguous()
    v    = v.contiguous()

    B = tmp7.shape[0]
    W = tmp7.shape[1]
    N = W * W

    soft = torch.empty((B, N, N), dtype=in0.dtype, device=in0.device)
    _rel_pos_softmax_H8_kernel[(B * N,)](
        tmp7, in2, in0, soft, B,
        tmp7.stride(0), tmp7.stride(1), tmp7.stride(3), tmp7.stride(4),
        in2.stride(0),  in2.stride(1),  in2.stride(2),
        in0.stride(0),  in0.stride(1),
        soft.stride(0), soft.stride(1),
    )
    return (soft @ v).transpose(-1, -2)


def replacement_func():
    return fused_rel_attn_H8