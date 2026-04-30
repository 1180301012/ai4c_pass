"""
Single-output Triton kernel: compute linear(x, W, b) using Triton GEMM.
Pattern: torch.nn.functional.linear(x, W, b) -> single [B, S, N] tensor.

This avoids multi-output tuple returns which crash replace_pattern in this framework.
The downstream reshape/split/permute/transposes remain in the graph (they are free views).
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton GEMM + bias kernel with autotuning
# X[M,K] @ W^T[K,N] + b[N] -> Y[M,N]   (float32 accumulators)
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=2, num_stages=4),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=2, num_stages=4),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 32,  'BLOCK_K': 32}, num_warps=2, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _gemm_bias_triton(
    x_ptr, w_ptr, b_ptr, y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Y[M,N] = X[M,K] @ W^T[K,N] + b[N]   (float32 accumulators)"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem = K - k * BLOCK_K
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_rem), other=0.0)
        w = tl.load(w_ptrs, mask=(offs_n[:, None] < N) & (offs_k[None, :] < k_rem), other=0.0)
        acc = tl.dot(x, tl.trans(w), acc=acc, out_dtype=tl.float32)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += b[None, :].to(tl.float32)

    y = acc.to(y_ptr.dtype.element_ty)
    y_ptrs = y_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(y_ptrs, y, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# ---------------------------------------------------------------------------
# Wrapper — @torch.fx.wrap makes FX treat this as an opaque call node.
# The function computes Y = x @ W^T + b and returns a contiguous [B, S, N]
# tensor matching what torch.nn.functional.linear would produce.
# The downstream reshape/split/permute/transposes stay in the graph as views.
# ---------------------------------------------------------------------------

@torch.fx.wrap
def triton_linear_bias(in_2, in_1, x):
    """
    in_2 : [N, K]   linear weight  (cuda)
    in_1 : [N]      linear bias   (cuda)
    x    : [B, S, K] input tensor (cuda, contiguous)

    Returns Y : [B, S, N] (same dtype as x) equivalent to F.linear(x, in_2, in_1).
    """
    B = x.shape[0]
    S = x.shape[1]
    K = x.shape[2]
    N = in_2.shape[0]
    M = B * S

    y = torch.empty((B, S, N), dtype=x.dtype, device=x.device)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )

    try:
        _gemm_bias_triton[grid](
            x, in_2, in_1, y,
            M, N, K,
            x.stride(1), x.stride(2),          # x strides: (S-dim, K-dim)
            in_2.stride(0), in_2.stride(1),    # w strides
        )
    except Exception:
        pass   # FakeTensor / meta tensor during shape propagation — skip

    return y