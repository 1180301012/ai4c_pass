"""
FuseRelAttnH16.py
Matches the full relative-position attention tail for W=16 (256×256 attention):
  tmp7.expand(-1,-1,16,-1,-1) → permute → +in2 → reshape(4,256,256) → +in0
  → softmax → @v → transpose(-1,-2)

Replacement:
  Kernel 1: Triton — fuses expand+permute+add(in2)+reshape+add(in0)+softmax
            → writes softmax [B, N, N]  (avoids 2 intermediate materializations)
  Kernel 2: Triton tiled GEMM — [B,N,N] @ [B,N,D] → [B,D,N]  (transposed output)
"""

import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────────────────────
# Pattern  (W=16, N=256)
# ─────────────────────────────────────────────────────────────────────────────

def pattern(tmp7, in2, in0, v):
    tmp_8  = tmp7.expand(-1, -1, 16, -1, -1)
    tmp_9  = tmp_8.permute((0, 3, 1, 4, 2))
    tmp_10 = tmp_9 + in2
    tmp_11 = tmp_10.reshape(4, 256, 256)
    tmp_12 = in0 + tmp_11
    tmp_13 = tmp_12.softmax(dim=-1)
    matmul_1 = tmp_13 @ v
    tmp_15 = matmul_1.transpose(-1, -2)
    return tmp_15


def replacement_args(tmp7, in2, in0, v):
    return (tmp7, in2, in0, v)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 1: fused rel-pos pre-processing + softmax  (one row per program)
#
# att[b,i,j] = in0[b,i,j]
#            + tmp7[b, i%W, 0, i//W, j//W]   ← broadcast across j%W
#            + in2[b,  i//W, i%W, j//W, j%W]
# out[b,i,:] = softmax(att[b,i,:])
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=2),
    ],
    key=['B'],
)
@triton.jit
def _rel_pos_softmax_H16_kernel(
    TMP7_ptr, IN2_ptr, IN0_ptr, OUT_ptr,
    B,
    s7b, s7h, s7p, s7k,   # tmp7 strides  [B, W, 1, W, W]
    s2b, s2r, s2c,         # in2  strides  [B, W, W, W, W]  (last 2 dims: W,1)
    s0b, s0i,              # in0  strides  [B, N, N]         (last dim: 1)
    sob, soi,              # out  strides  [B, N, N]
    W: tl.constexpr = 16,
    N: tl.constexpr = 256,
):
    pid = tl.program_id(0)
    b   = pid // N
    i   = pid %  N

    q_row = i // W
    q_col = i %  W

    offs_kr = tl.arange(0, W)
    offs_kc = tl.arange(0, W)
    offs_j  = offs_kr[:, None] * W + offs_kc[None, :]   # [W, W] j-indices

    in0_row   = tl.load(IN0_ptr  + b*s0b + i*s0i  + offs_j         ).to(tl.float32)
    tmp7_vals = tl.load(TMP7_ptr + b*s7b + q_col*s7h + q_row*s7p + offs_kr*s7k).to(tl.float32)
    in2_row   = tl.load(IN2_ptr  + b*s2b + q_row*s2r + q_col*s2c + offs_j).to(tl.float32)

    att     = in0_row + tmp7_vals[:, None] + in2_row        # [W, W]
    row_max = tl.max(att, axis=1)
    att_max = tl.max(row_max, axis=0)
    exp_att = tl.exp(att - att_max)
    exp_sum = tl.sum(tl.sum(exp_att, axis=1), axis=0)
    soft    = (exp_att / exp_sum).to(TMP7_ptr.dtype.element_ty)

    tl.store(OUT_ptr + b*sob + i*soi + offs_j, soft)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 2: tiled batched GEMM  A[B,M,N] @ B[B,N,D] → Out[B,D,M]
#   (output is stored in transposed layout to avoid a separate transpose kernel)
#   tl.dot constraints: BM ≥ 16, BK ≥ 16, BD ≥ 16.
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BM': 16, 'BK': 64, 'BD': 128}, num_warps=4, num_stages=2),
        triton.Config({'BM': 32, 'BK': 64, 'BD': 128}, num_warps=4, num_stages=2),
        triton.Config({'BM': 16, 'BK': 32, 'BD': 128}, num_warps=4, num_stages=2),
        triton.Config({'BM': 32, 'BK': 32, 'BD': 128}, num_warps=4, num_stages=2),
        triton.Config({'BM': 16, 'BK': 64, 'BD': 128}, num_warps=8, num_stages=2),
        triton.Config({'BM': 32, 'BK': 64, 'BD': 128}, num_warps=8, num_stages=2),
        triton.Config({'BM': 16, 'BK': 64, 'BD': 64},  num_warps=4, num_stages=2),
    ],
    key=['M', 'N', 'D'],
)
@triton.jit
def _batched_mm_T_H16(
    A_ptr, B_ptr, Out_ptr,
    M, N, D,
    sAb, sAm, sAn,
    sBb, sBn, sBd,
    sOb, sOd, sOm,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BD: tl.constexpr,
):
    b     = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_d = tl.program_id(2)

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_d = pid_d * BD + tl.arange(0, BD)
    mask_m = offs_m < M
    mask_d = offs_d < D

    acc = tl.zeros((BM, BD), dtype=tl.float32)

    for k0 in range(0, N, BK):
        offs_k = k0 + tl.arange(0, BK)
        mask_k = offs_k < N

        a_tile = tl.load(
            A_ptr + b*sAb + offs_m[:, None]*sAm + offs_k[None, :]*sAn,
            mask=(mask_m[:, None] & mask_k[None, :]), other=0.0,
        )
        b_tile = tl.load(
            B_ptr + b*sBb + offs_k[:, None]*sBn + offs_d[None, :]*sBd,
            mask=(mask_k[:, None] & mask_d[None, :]), other=0.0,
        )
        acc += tl.dot(a_tile, b_tile)

    # transposed store: Out[b, d, m]
    out_ptrs = Out_ptr + b*sOb + offs_d[:, None]*sOd + offs_m[None, :]*sOm
    tl.store(out_ptrs, tl.trans(acc).to(A_ptr.dtype.element_ty),
             mask=(mask_d[:, None] & mask_m[None, :]))


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper
# ─────────────────────────────────────────────────────────────────────────────

@torch.fx.wrap
def fused_rel_attn_H16(tmp7, in2, in0, v):
    tmp7 = tmp7.contiguous()
    in2  = in2.contiguous()
    in0  = in0.contiguous()
    v    = v.contiguous()

    B = tmp7.shape[0]
    W = tmp7.shape[1]   # 16
    N = W * W           # 256
    D = v.shape[-1]     # 128

    # --- kernel 1: fused rel-pos preprocessing + softmax ---
    soft = torch.empty((B, N, N), dtype=in0.dtype, device=in0.device)
    _rel_pos_softmax_H16_kernel[(B * N,)](
        tmp7, in2, in0, soft, B,
        tmp7.stride(0), tmp7.stride(1), tmp7.stride(3), tmp7.stride(4),
        in2.stride(0),  in2.stride(1),  in2.stride(2),
        in0.stride(0),  in0.stride(1),
        soft.stride(0), soft.stride(1),
    )

    # --- stage 2: batched GEMM via native @ (cuBLAS) + transpose view ---
    # soft [B, N, N] @ v [B, N, D] → [B, N, D] → transpose → [B, D, N]
    return (soft @ v).transpose(-1, -2)


def replacement_func():
    return fused_rel_attn_H16