"""
Shared dispatch module for all Triton matmul kernel passes.
Both FusedAttnMatmulView20x20 and FusedMatmulView1x1 import this and return
the SAME dispatch_kernel object, satisfying the replacement_func_limit.

Route strings:
  "route_20x20"  → batched GEMM, view(B, H*M, 20, 20)
  "route_1x1"    → batched GEMV, view(B, C1, 1, 1)
"""
import torch
import triton
import triton.language as tl
import math


# ─────────────────────────────────────────────────────────────────────────────
# Kernel A: Batched GEMM for attention 20×20 pattern
#   in_1 [B, H, M, K]  @  in_0 [B, H, K, N]  →  out [B, H*M, N, N]
#   K = N = 400,  N = sqrt(in_0.shape[3])
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=2, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _batched_gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    a_sb, a_sh, a_sm, a_sk,
    b_sb, b_sh, b_sk, b_sn,
    c_sb, c_sh, c_sm, c_sn,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Batched GEMM: C[bh, m, n] = A[bh, m, :] · B[bh, :, n]"""
    bh = tl.program_id(0)
    rm = tl.program_id(1)
    rn = tl.program_id(2)

    rm_off = rm * BLOCK_M + tl.arange(0, BLOCK_M)
    rn_off = rn * BLOCK_N + tl.arange(0, BLOCK_N)

    a_base = a_ptr + bh * a_sb
    b_base = b_ptr + bh * b_sb
    c_base = c_ptr + bh * c_sb

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)
        a = tl.load(
            a_base + rm_off[:, None] * a_sm + rk[None, :] * a_sk,
            mask=(rm_off[:, None] < M) & (rk[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_base + rk[:, None] * b_sk + rn_off[None, :] * b_sn,
            mask=(rk[:, None] < K) & (rn_off[None, :] < N),
            other=0.0,
        )
        acc = tl.dot(a, b, acc)

    tl.store(
        c_base + rm_off[:, None] * c_sm + rn_off[None, :] * c_sn,
        acc.to(c_ptr.dtype.element_ty),
        mask=(rm_off[:, None] < M) & (rn_off[None, :] < N),
    )


def _run_batched_gemm_20x20(in_0, in_1):
    """
    Computes in_1 @ in_0 → [B, H*M, N, N] (avoids extra view call).
      in_1 [B, H, M, K],  in_0 [B, H, K, 400]
    """
    B  = in_1.shape[0]
    H  = in_1.shape[1]
    M  = in_1.shape[2]
    K  = in_1.shape[3]
    N  = int(math.sqrt(in_0.shape[3]))   # = 400 for all attention graphs

    out = torch.empty((B, H * M, N, N), dtype=in_1.dtype, device=in_1.device)

    BH = B * H
    grid = lambda meta: (
        BH,
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )

    _batched_gemm_kernel[grid](
        in_1, in_0, out,
        in_1.stride(0), in_1.stride(1), in_1.stride(2), in_1.stride(3),
        in_0.stride(0), in_0.stride(1), in_0.stride(2), in_0.stride(3),
        out.stride(0),  out.stride(1),  out.stride(2),  out.stride(3),
        M, N, K,
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Kernel B: Batched GEMV for 1×1 view pattern  (N=1 in second operand)
#   in_1 [B, 1, M, K]  @  in_0 [B, 1, K, 1]  →  out [B, 1, M, 1]
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_K': 64},  num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 64},  num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_K': 128}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 128}, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_K': 64},  num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_K': 128}, num_warps=8),
        triton.Config({'BLOCK_M': 32,  'BLOCK_K': 64},  num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_K': 128}, num_warps=4),
    ],
    key=['M', 'K'],
)
@triton.jit
def _batched_gemv_kernel(
    a_ptr, b_ptr, c_ptr,
    a_sb, a_sm, a_sk,
    b_sb, b_sk,
    c_sb, c_sm,
    M, K,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Batched GEMV (N=1):
      out[b, 0, m, 0] = Σ_k  in_1[b, 0, m, k] * in_0[b, 0, k, 0]
    """
    bb = tl.program_id(0)
    rm = tl.program_id(1)

    rm_off = rm * BLOCK_M + tl.arange(0, BLOCK_M)
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    a_base = a_ptr + bb * a_sb
    b_base = b_ptr + bb * b_sb
    c_base = c_ptr + bb * c_sb

    for k in range(0, K, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)
        a_tile = tl.load(
            a_base + rm_off[:, None] * a_sm + rk[None, :] * a_sk,
            mask=(rm_off[:, None] < M) & (rk[None, :] < K),
            other=0.0,
        )
        b_col = tl.load(
            b_base + rk * b_sk,
            mask=rk < K,
            other=0.0,
        )
        acc += tl.sum(
            a_tile.to(tl.float32) * b_col[None, :],
            axis=1,
        )

    tl.store(
        c_base + rm_off * c_sm,
        acc.to(c_ptr.dtype.element_ty),
        mask=rm_off < M,
    )


def _run_batched_gemv_1x1(in_0, in_1):
    """
    Computes torch.matmul(in_1, in_0) → [B, 1, M, 1].
      in_1 [B, 1, M, K],  in_0 [B, 1, K, 1]
    """
    B  = in_1.shape[0]
    M  = in_1.shape[2]
    K  = in_1.shape[3]

    out = torch.empty((B, 1, M, 1), dtype=in_1.dtype, device=in_1.device)

    grid = lambda meta: (B, triton.cdiv(M, meta['BLOCK_M']))

    _batched_gemv_kernel[grid](
        in_1, in_0, out,
        in_1.stride(0), in_1.stride(2), in_1.stride(3),
        in_0.stride(0), in_0.stride(2),
        out.stride(0),  out.stride(2),
        M, K,
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Shared dispatch wrapper  (SAME object imported by every pass file)
# ─────────────────────────────────────────────────────────────────────────────

@torch.fx.wrap
def dispatch_kernel(in_0, in_1, route):
    if route == "route_20x20":
        return _run_batched_gemm_20x20(in_0, in_1)
    elif route == "route_1x1":
        return _run_batched_gemv_1x1(in_0, in_1)