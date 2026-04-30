"""
Pass: FusedLinearMul_rtmpose

Matches the pattern:
    linear = torch.nn.functional.linear(in_3, in_0, None)
    tmp_3  = in_2 * in_1
    return (tmp_3, linear)

in_0: weight  [N, K]      e.g. [256, 512]
in_1: scale   [N]         e.g. [256]
in_2: factor  [..., N]    e.g. [B, S, 256]
in_3: input   [..., K]    e.g. [B, S, 512]

Strategy:
  - linear output  (in_3 @ in_0.T) : two Triton kernels:
      1. pure-GEMM kernel (specialised for no-scale case) via the shared module
      2. broadcast-multiply kernel (new, module-level in this file)
"""

import torch
import triton
import triton.language as tl

from pass_dir.fused_linear_scale_mul import fused_linear_scale_mul_kernel


# ---------------------------------------------------------------------------
# Kernel 1: pure GEMM (no scale multiply)
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 32, 'num_stages': 2, 'num_warps': 2}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32,  'BLOCK_K': 32, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 32, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64, 'num_stages': 4, 'num_warps': 8}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _pure_gemm_kernel(
    x_ptr, w_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Computes out = x @ w.T  (no scaling)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)

        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)

        w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
        w_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc += tl.dot(x, tl.trans(w))

    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=out_mask)


# ---------------------------------------------------------------------------
# Kernel 2: broadcast element-wise multiply
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 512},  num_warps=2),
        triton.Config({'BLOCK': 1024}, num_warps=4),
        triton.Config({'BLOCK': 2048}, num_warps=4),
        triton.Config({'BLOCK': 4096}, num_warps=8),
    ],
    key=['n_elements', 'n_cols'],
)
@triton.jit
def _broadcast_mul_kernel(
    in2_ptr, in1_ptr, out_ptr,
    n_elements, n_cols,
    BLOCK: tl.constexpr,
):
    """out[i] = in2[i] * in1[i // n_cols]"""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    row = offs // n_cols
    val = tl.load(in2_ptr + offs, mask=mask, other=0.0)
    val = val * tl.load(in1_ptr + row % n_cols, mask=mask, other=1.0)
    tl.store(out_ptr + offs, val, mask=mask)


# ---------------------------------------------------------------------------
# Pattern / replacement
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2, in_3):
    linear = torch.nn.functional.linear(in_3, in_0, None)
    tmp_3 = in_2 * in_1
    return (tmp_3, linear)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@torch.fx.wrap
def fused_linear_mul_rtmpose(in_0, in_1, in_2, in_3):
    """
    in_0 : weight  [N, K]
    in_1 : scale   [N]
    in_2 : factor  [..., N]
    in_3 : input   [..., K]

    Returns (tmp_3, linear) where:
        linear = in_3 @ in_0.T          (pure GEMM)
        tmp_3  = in_2 * in_1            (broadcast scale)
    """
    # ---- Part A: linear = in_3 @ in_0.T (pure GEMM) ----
    leading = in_3.shape[:-1]
    K = in_3.shape[-1]
    N = in_0.shape[0]
    M = in_3.numel() // K

    in3_2d = in_3.reshape(M, K)
    in0_2d = in_0.reshape(N, K)
    out_linear_2d = torch.empty((M, N), dtype=in_3.dtype, device=in_3.device)

    grid_a = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),
                           triton.cdiv(N, meta['BLOCK_N']))
    _pure_gemm_kernel[grid_a](
        in3_2d, in0_2d, out_linear_2d,
        M, N, K,
        in3_2d.stride(0), in3_2d.stride(1),
        in0_2d.stride(0), in0_2d.stride(1),
    )
    out_linear = out_linear_2d.reshape(*leading, N)

    # ---- Part B: tmp_3 = in_2 * in_1 (broadcast multiply) ----
    n_elements = in_2.numel()
    n_cols     = in_1.numel()
    tmp_3_out  = torch.empty_like(in_2)

    grid_b = lambda meta: (triton.cdiv(n_elements, meta['BLOCK']),)
    _broadcast_mul_kernel[grid_b](
        in_2, in_1, tmp_3_out,
        n_elements, n_cols,
    )

    return (tmp_3_out, out_linear)


def replacement_func():
    return fused_linear_mul_rtmpose