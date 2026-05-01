"""
Shared Triton matmul kernel + wrapper for fast QKV linear projection.

Replaces: torch.nn.functional.linear(x, w, None)   x:[1,M,K]  w:[N,K]
With:     _fast_linear(w, x)  returning [1, M, N] via a Triton tiled matmul.

The downstream reshape / permute / unbind / transpose are view-ops (free)
and remain untouched in the graph.

M=197 for all graphs; K=192/432/768; N=576/1296/2304.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton tiled matmul:  out[m, n] = sum_k  x[m, k] * w[n, k]   (x @ w^T)
# Fixed config (no autotune) to avoid autotuning overhead spilling into
# the 100-trial benchmark window.
# BLOCK_M=64, BLOCK_N=64, BLOCK_K=64 works well for all graph sizes.
# ---------------------------------------------------------------------------
@triton.jit
def _matmul_kernel(
    x_ptr, w_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = rm < M
    mask_n = rn < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        rk = k_start * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = rk < K

        # x tile [BLOCK_M, BLOCK_K] — row-major, coalesced in k
        x_tile = tl.load(
            x_ptr + rm[:, None] * stride_xm + rk[None, :] * stride_xk,
            mask=mask_m[:, None] & k_mask[None, :],
            other=0.0,
        )
        # w tile [BLOCK_N, BLOCK_K] — row-major, coalesced in k
        w_tile = tl.load(
            w_ptr + rn[:, None] * stride_wn + rk[None, :] * stride_wk,
            mask=mask_n[:, None] & k_mask[None, :],
            other=0.0,
        )
        # acc += x_tile @ w_tile^T : [BLOCK_M,BLOCK_K] @ [BLOCK_K,BLOCK_N]
        acc += tl.dot(x_tile, tl.trans(w_tile))

    out_ptrs = out_ptr + rm[:, None] * N + rn[None, :]
    tl.store(
        out_ptrs,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_n[None, :],
    )


# ---------------------------------------------------------------------------
# @torch.fx.wrap wrapper — SINGLE output, easy 1:1 pattern replacement
# ---------------------------------------------------------------------------
@torch.fx.wrap
def _fast_linear(w, x):
    """
    w : weight  [N, K]       (in_0 from pattern)
    x : input   [1, M, K]    (in_1 from pattern)
    Returns out [1, M, N]  = x @ w^T  (same as F.linear(x, w, None))
    """
    M = x.shape[1]
    K = x.shape[2]
    N = w.shape[0]

    out = torch.empty((1, M, N), dtype=x.dtype, device=x.device)

    BM, BN, BK = 64, 64, 64
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _matmul_kernel[grid](
        x, w, out,
        M, N, K,
        x.stride(1), x.stride(2),
        w.stride(0), w.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
        num_stages=4, num_warps=4,
    )
    return out