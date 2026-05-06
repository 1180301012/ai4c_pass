"""
Shared Triton kernels + unified dispatch for MatMulScale passes.

Provides:
  - fused_dispatch(arg0, arg1, arg2, route STRING)
      arg0, arg1, arg2: tensors (arg2 is always the 'tail' / slowdown tensor)
      route: "gemm" | "smol" | "elem"

  - matmul_scale_kernel(x, weight, scale, out, M, N, K)
      Fused GEMM: out = (x @ weight^T) * scale[i] per output feature i
  - flat_scale_kernel(x, scale, out, N, N_scale)
      Elementwise: out[i] = x[i] * scale[i % N_scale]
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel: fused linear (matmul) + per-feature elementwise scale.
#
# out[m, n] = (sum_k x[m, k] * w[n, k]) * scale[n]
# Grid: (ceil(M/BLOCK_M), ceil(N/BLOCK_N))
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N':  64, 'BLOCK_K':  64, 'num_warps': 4, 'num_stages': 2}),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K':  64, 'num_warps': 4, 'num_stages': 2}),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K':  64, 'num_warps': 4, 'num_stages': 2}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N':  64, 'BLOCK_K':  64, 'num_warps': 4, 'num_stages': 2}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K':  64, 'num_warps': 8, 'num_stages': 2}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K':  64, 'num_warps': 8, 'num_stages': 2}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N':  64, 'BLOCK_K': 128, 'num_warps': 4, 'num_stages': 2}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 128, 'num_warps': 8, 'num_stages': 2}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N':  64, 'BLOCK_K':  64, 'num_warps': 4, 'num_stages': 2}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K':  64, 'num_warps': 8, 'num_stages': 2}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K':  64, 'num_warps': 8, 'num_stages': 2}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 128, 'num_warps': 8, 'num_stages': 2}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K':  64, 'num_warps': 8, 'num_stages': 3}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'num_warps': 8, 'num_stages': 3}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_mul_scale_kernel(
    x_ptr, w_ptr, scale_ptr, out_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused GEMM + elementwise scale.
    Computes: out = (X @ W^T) * scale  where scale has length N.
    All tensors stored in row-major (C-contiguous).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulate in fp32 for numerical precision
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_tile in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k_tile * BLOCK_K + tl.arange(0, BLOCK_K)

        # Load x tile [BLOCK_M, BLOCK_K]
        x_ptrs = x_ptr + offs_m[:, None] * K + offs_k[None, :]
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # Load w tile [BLOCK_N, BLOCK_K]  (weight stored as [N, K])
        w_ptrs = w_ptr + offs_n[:, None] * K + offs_k[None, :]
        w_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # acc += x_tile @ w_tile^T  →  [BLOCK_M, BLOCK_N]
        acc = tl.dot(x_tile, tl.trans(w_tile), acc, out_dtype=tl.float32)

    # Load per-feature scale [BLOCK_N]
    scale_vals = tl.load(scale_ptr + offs_n, mask=offs_n < N, other=1.0)

    # out[b,m,n] = acc * scale[n]
    result = acc * scale_vals[None, :]

    # Store output (cast back to original dtype)
    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, result, mask=out_mask)


# ---------------------------------------------------------------------------
# Kernel: flat elementwise multiply with broadcasting.
#
# For each block of BLOCK_K elements:
#   out[i] = x[i] * scale[i % N_scale]
# Grid: (ceil(N / BLOCK_K),)
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_K':  512, 'num_warps': 4, 'num_stages': 2}),
        triton.Config({'BLOCK_K': 1024, 'num_warps': 4, 'num_stages': 2}),
        triton.Config({'BLOCK_K': 2048, 'num_warps': 8, 'num_stages': 2}),
        triton.Config({'BLOCK_K': 4096, 'num_warps': 8, 'num_stages': 2}),
    ],
    key=['N', 'N_scale'],
)
@triton.jit
def flat_mul_kernel(
    x_ptr, scale_ptr, out_ptr,
    N, N_scale,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = offs < N
    scale_idx = offs % N_scale          # broadcast scale over the N dimension

    x   = tl.load(x_ptr   + offs, mask=mask)
    scale = tl.load(scale_ptr + scale_idx, mask=mask)

    tl.store(out_ptr + offs, x * scale, mask=mask)


# ---------------------------------------------------------------------------
# Kernel: flat elementwise multiply with per-batch scale (for elementwise-
# pass inputs that are already the same shape; used by ElemwiseMul).
#
# For each block of BLOCK_K elements:
#   out[i] = x[i] * scale[i % N_scale]
# Grid: (ceil(N / BLOCK_K),)
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_K': 1024, 'num_warps': 4,  'num_stages': 2}),
        triton.Config({'BLOCK_K': 2048, 'num_warps': 4,  'num_stages': 2}),
        triton.Config({'BLOCK_K': 4096, 'num_warps': 8,  'num_stages': 2}),
        triton.Config({'BLOCK_K': 8192, 'num_warps': 8,  'num_stages': 3}),
        triton.Config({'BLOCK_K': 16384,'num_warps': 8,  'num_stages': 3}),
    ],
    key=['N', 'N_scale'],
)
@triton.jit
def flat_scale_kernel(
    x_ptr, scale_ptr, out_ptr,
    N, N_scale,
    BLOCK_K: tl.constexpr,
):
    pid  = tl.program_id(0)
    offs = pid * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = offs < N
    scale_idx = offs % N_scale          # broadcast scale over the N dimension

    x   = tl.load(x_ptr   + offs, mask=mask)
    scale = tl.load(scale_ptr + scale_idx, mask=mask)

    tl.store(out_ptr + offs, x * scale, mask=mask)


# ---------------------------------------------------------------------------
# Shared dispatch wrapper used by ALL MatMulScale passes.
# Route string is the last arg; routes are "gemm", "smol", "elem".
#
# "gemm"   : fused GEMM+scale   arg0=weight[N,K] arg1=scale[N]  arg2=x[M,K flat]
# "smol"   : same kernel         arg0=weight[N,K] arg1=scale[M,N] arg2=x[M,K flat]
# "elem"   : pure broadcast mul  arg0=x[M,N flat]       arg1=scale[N] arg2=ignored
#
# NOTE: Tensors are passed WITHOUT reshape/view.  Triton kernels accept raw
#       data_ptrs and work on flat [M*N] views.  C-contiguous assumption.
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_dispatch(arg0, arg1, arg2, route):
    if route == "gemm":
        # arg0=weight[N,K], arg1=scale[N], arg2=x[M*K elements as 1D]
        N_out = arg0.shape[0]
        K     = arg0.shape[1]
        total = arg2.numel()                          # = M * K
        M     = total // K
        out   = torch.empty_like(arg2)               # same shape as x
        grid  = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),
                               triton.cdiv(N_out, meta['BLOCK_N']))
        gemm_mul_scale_kernel[grid](arg2, arg0, arg1, out, M, N_out, K)
        return out

    elif route == "smol":
        # arg0=weight[N,K], arg1=scale[M,N same shape], arg2=x[M*K flat]
        N_out = arg0.shape[0]
        K     = arg0.shape[1]
        total = arg2.numel()
        M     = total // K
        out   = torch.empty_like(arg1)               # same shape as scale
        grid  = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),
                               triton.cdiv(N_out, meta['BLOCK_N']))
        gemm_mul_scale_kernel[grid](arg2, arg0, arg1, out, M, N_out, K)
        return out

    elif route == "elem":
        # arg0=x[M*N], arg1=scale[N], arg2=ignored
        N     = arg0.numel()
        N_sc  = arg1.numel()
        out   = torch.empty_like(arg0)               # same shape
        grid  = lambda meta: (triton.cdiv(N, meta['BLOCK_K']),)
        flat_scale_kernel[grid](arg0, arg1, out, N, N_sc)
        return out

    # never reached
    return arg0