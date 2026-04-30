"""
Fused batched-matmul + transpose + reshape kernel.
Pattern matches:  matmul(attn, in_2) → transpose(1,2) → contiguous → reshape(1,257,-1) → contiguous
Works for bf16, fp16, fp32 models with the same graph structure.

Inputs:
  attn  [B, H, M, N]  – attention weights (bf16/fp16/fp32)
  in_2  [B, H, N, D]  – value states  (bf16/fp16/fp32)
  output: [B, M, H*D]  (same dtype as attn)
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
    ],
    key=['M', 'N', 'D'],
)
@triton.jit
def batched_matmul_transpose_kernel(
    A_ptr, B_ptr, C_ptr,
    stride_ab, stride_ah, stride_am, stride_an,
    stride_bb, stride_bh, stride_bn, stride_bd,
    stride_cb, stride_cm,
    B, H, M, N, D,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,   # = 80 (= 16*5, divisible by 16 for tl.dot)
):
    """
    Each program handles one (b*h, query-block) pair.
    Grid: (B*H, ceil(M/BLOCK_M)). Computes C[b,m,h*80+d] = sum_k A*B.
    BLOCK_D=80 accepted by tl.dot since 80 % 16 == 0.
    """
    bh = tl.program_id(0)
    bm = tl.program_id(1)

    b  = bh // H
    h  = bh % H

    off_m = bm * BLOCK_M + tl.arange(0, BLOCK_M)
    off_d = tl.arange(0, 128)   # BLOCK_D=128 (next power-of-2 >= 80)
    mask_d = off_d < D           # mask out padding beyond actual D=80

    acc = tl.zeros([BLOCK_M, 128], dtype=tl.float32)

    A_bh = A_ptr + b * stride_ab + h * stride_ah
    B_bh = B_ptr + b * stride_bb + h * stride_bh

    for k in range(0, tl.cdiv(N, BLOCK_K)):
        off_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = off_k < N

        # Load A[b,h,off_m,:] — attn block [BLOCK_M, BLOCK_K]
        a = tl.load(
            A_bh + off_m[:, None] * stride_am + off_k[None, :] * stride_an,
            mask=off_m[:, None] < M,
            other=0.0,
        ).to(tl.float32)

        # Load B[b,h,off_k,:] — value block [BLOCK_K, 128]
        b_blk = tl.load(
            B_bh + off_k[:, None] * stride_bn + off_d[None, :] * stride_bd,
            mask=mask_k[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)

        acc = acc + tl.dot(a, b_blk, allow_tf32=True)

    # Write C[b, off_m, h*D + off_d]   — out layout [B, M, H*D]
    C_bh   = C_ptr + b * stride_cb
    mask_m = off_m < M
    tl.store(
        C_bh + off_m[:, None] * stride_cm + h * D + off_d[None, :],
        acc,
        mask=mask_m[:, None] & mask_d[None, :],
    )


@torch.fx.wrap
def matmul_transpose_reshape(attn, in_2):
    """
    attn:  [B, H, M, N]  (bf16 / fp16 / fp32)
    in_2:  [B, H, N, D]  (same dtype)
    Returns: [B, M, H*D]  (same dtype, fused transpose+reshape)
    """
    B, H, M, N = attn.shape
    _, _,  _, D = in_2.shape

    out = torch.empty(B, M, H * D, dtype=attn.dtype, device=attn.device)

    grid = lambda meta: (B * H, triton.cdiv(M, meta['BLOCK_M']))

    batched_matmul_transpose_kernel[grid](
        attn, in_2, out,
        attn.stride(0),    attn.stride(1),    attn.stride(2),    attn.stride(3),
        in_2.stride(0),    in_2.stride(1),    in_2.stride(2),    in_2.stride(3),
        out.stride(0),     out.stride(1),
        B, H, M, N, D,
        BLOCK_D=128,
    )
    return out