"""
Optimization pass for: dropout(x, p=0.0, training=False) + .to(bf16) + linear(x, w, b)  [bfloat16]

Covers: RECT_L (bfloat16)
Input shape: [128, 128] -> output [128, 128]
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern to match
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2):
    tmp_2 = torch.nn.functional.dropout(in_2, p=0.0, training=False)
    to = tmp_2.to(torch.bfloat16)
    linear = torch.nn.functional.linear(to, in_1, in_0)
    return linear


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# ---------------------------------------------------------------------------
# Triton GEMM + bias kernel  (bf16 dtype)
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_stages=1, num_warps=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'BLOCK_K': 32}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _fused_linear_cast_bf16(
    x_ptr,       # [M, K]  bf16
    w_ptr,       # [N, K]  bf16  (weight: each row = output neuron weights)
    b_ptr,       # [N]     bf16  (bias)
    out_ptr,     # [M, N]  bf16
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)

        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        w_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)

        x = tl.load(x_ptr + offs_m[:, None] * K + offs_k[None, :],
                    mask=x_mask, other=0.0)
        w = tl.load(w_ptr + offs_n[:, None] * K + offs_k[None, :],
                    mask=w_mask, other=0.0)

        # acc[m, n] += dot(x[m,:], w[n,:])^T
        acc = tl.dot(x.to(tl.float32), tl.trans(w.to(tl.float32)), acc)

    # add bias
    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc += b[None, :]

    # cast back to bf16 and store
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :],
             acc.to(tl.bfloat16), mask=out_mask)


# ---------------------------------------------------------------------------
# Python wrapper  (must be @torch.fx.wrap)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def _fused_dropout_cast_linear_bf16(in_0, in_1, in_2):
    """
    in_0 : bias   [out_features]
    in_1 : weight [out_features, in_features]
    in_2 : input  [*, in_features]  (already on CUDA in bf16 for RECT_L)
    """
    K = in_2.shape[-1]
    M = in_2.numel() // K
    N = in_1.shape[0]

    # Reshape input to 2-D for the kernel
    x_flat = in_2.view(M, K)

    out_flat = torch.empty((M, N), dtype=torch.bfloat16, device=in_2.device)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(N, META['BLOCK_N']),
    )

    _fused_linear_cast_bf16[grid](
        x_ptr=x_flat,
        w_ptr=in_1,
        b_ptr=in_0,
        out_ptr=out_flat,
        M=M, N=N, K=K,
    )

    # Restore batch/sequence dimensions
    orig_shape = list(in_2.shape[:-1]) + [N]
    return out_flat.view(orig_shape)


# ---------------------------------------------------------------------------
# replacement_func
# ---------------------------------------------------------------------------
def replacement_func():
    return _fused_dropout_cast_linear_bf16