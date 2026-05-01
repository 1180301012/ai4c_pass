"""
Optimization pass for BotNet attention – fully Triton-based:
  1. Kernel 1: fused (x + y) + row-wise softmax  →  attn [B, S, S]
  2. Kernel 2: batched matmul (attn @ mat) writing output transposed
                →  out [B, D, S]  (equivalent to .transpose(-1,-2))

Pattern matched:  (x + y).softmax(dim=-1) @ mat → .transpose(-1,-2)
Shape-agnostic: covers S=256 (16x16) and S=64 (8x8) variants for fp16/bf16.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern to match
# ---------------------------------------------------------------------------
def pattern(x, y, mat):
    combined = x + y
    attn = combined.softmax(dim=-1)
    out = attn @ mat
    result = out.transpose(-1, -2)
    return (result,)


def replacement_args(x, y, mat):
    return (x, y, mat)


# ---------------------------------------------------------------------------
# Kernel 1: fused  (x + y)  →  row-wise softmax
# Grid: (B * S,)   one program per row
# ---------------------------------------------------------------------------
@triton.jit
def add_softmax_kernel(
    x_ptr, y_ptr, out_ptr,
    n_cols,
    IS_FP16:    tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row  = tl.program_id(0)
    base = row * n_cols
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols

    xr = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    yr = tl.load(y_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    d  = xr + yr

    d_max = tl.max(d, axis=0)
    d     = d - d_max
    e     = tl.exp(d)
    s     = tl.sum(e, axis=0)
    sm    = e / s

    if IS_FP16:
        tl.store(out_ptr + base + offs, sm.to(tl.float16), mask=mask)
    else:
        tl.store(out_ptr + base + offs, sm.to(tl.bfloat16), mask=mask)


# ---------------------------------------------------------------------------
# Kernel 2: batched GEMM  C[b, n, m] = Σ_k  A[b, m, k] * B[b, k, n]
#   A = attn  [B, M, K]     M = S (rows),  K = S (cols)
#   B = mat   [B, K, N]     N = D
#   C = out   [B, N, M]     stored transposed  → same as (A@B).T  [B, D, S]
# Grid: (B, ceil(M/BM), ceil(N/BN))
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def batched_matmul_T_kernel(
    a_ptr, b_ptr, c_ptr,
    B, M, N, K,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cn, stride_cm,
    IS_FP16: tl.constexpr,
    BLOCK_M:  tl.constexpr,
    BLOCK_N:  tl.constexpr,
    BLOCK_K:  tl.constexpr,
):
    b     = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_off = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_off = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_off = k_start + tl.arange(0, BLOCK_K)

        # A[b, m, k]
        a_ptrs = a_ptr + b * stride_ab + m_off[:, None] * stride_am + k_off[None, :] * stride_ak
        a_mask = (m_off[:, None] < M) & (k_off[None, :] < K)
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # B[b, k, n]
        b_ptrs = b_ptr + b * stride_bb + k_off[:, None] * stride_bk + n_off[None, :] * stride_bn
        b_mask = (k_off[:, None] < K) & (n_off[None, :] < N)
        b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a_tile, b_tile, out_dtype=tl.float32)

    # Write C[b, n, m]  (transposed relative to A@B)
    # acc is [BLOCK_M, BLOCK_N]; c_ptrs is [BLOCK_N, BLOCK_M] → transpose acc
    c_ptrs = c_ptr + b * stride_cb + n_off[:, None] * stride_cn + m_off[None, :] * stride_cm
    c_mask = (n_off[:, None] < N) & (m_off[None, :] < M)
    acc_T  = tl.trans(acc)   # [BLOCK_N, BLOCK_M]

    if IS_FP16:
        tl.store(c_ptrs, acc_T.to(tl.float16), mask=c_mask)
    else:
        tl.store(c_ptrs, acc_T.to(tl.bfloat16), mask=c_mask)


# ---------------------------------------------------------------------------
# Wrapper  (must be @torch.fx.wrap)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_add_softmax_attn(x, y, mat):
    B, S, _  = x.shape     # e.g. (4, 256, 256)
    _, _, D  = mat.shape   # e.g. D = 128
    IS_FP16  = 1 if x.dtype == torch.float16 else 0

    # Next power-of-two >= S  (S ∈ {64, 256} here)
    BLOCK_SIZE = 1
    while BLOCK_SIZE < S:
        BLOCK_SIZE *= 2

    # Step 1 – fused add + softmax
    attn = torch.empty((B, S, S), dtype=x.dtype, device=x.device)
    add_softmax_kernel[(B * S,)](
        x, y, attn,
        n_cols=S,
        IS_FP16=IS_FP16,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Step 2 – batched GEMM writing output transposed  →  [B, D, S]
    out = torch.empty((B, D, S), dtype=x.dtype, device=x.device)

    # Use a lambda grid so autotune's chosen block sizes drive the grid
    grid = lambda meta: (
        B,
        triton.cdiv(S, meta['BLOCK_M']),
        triton.cdiv(D, meta['BLOCK_N']),
    )

    batched_matmul_T_kernel[grid](
        attn, mat, out,
        B, S, D, S,           # B, M=S, N=D, K=S
        S * S, S, 1,          # attn strides  [b, m, k]
        S * D, D, 1,          # mat  strides  [b, k, n]
        D * S, S, 1,          # out  strides  [b, n, m]
        IS_FP16=IS_FP16,
    )

    return (out,)


def replacement_func():
    return fused_add_softmax_attn