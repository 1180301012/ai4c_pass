"""
Shared Triton kernel: fused GEMM (linear) + element-wise scale multiply.

Computes: out[m, n] = (sum_k x[m,k] * w[n,k]) * scale[n]
  - x     : [M, K]  (input activations)
  - w     : [N, K]  (weight matrix, transposed for GEMM)
  - scale : [N]     (scale factor per output channel)
  - out   : [M, N]  (result)

This fuses the GEMM output write with the scale multiplication,
avoiding an extra global-memory round-trip for the intermediate tensor.
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 32,  'BLOCK_K': 32, 'num_stages': 2, 'num_warps': 2}),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 32, 'num_stages': 2, 'num_warps': 2}),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 32, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32,  'BLOCK_K': 32, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64, 'num_stages': 4, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 64,  'BLOCK_K': 64, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 64, 'num_stages': 3, 'num_warps': 4}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_linear_scale_mul_kernel(
    x_ptr, w_ptr, scale_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Each program block computes a BLOCK_M x BLOCK_N tile of the output.
    After accumulating the GEMM, multiplies by the per-column scale vector.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # [BLOCK_N]

    # Accumulator in fp32 for numerical stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over the K (reduction) dimension
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)   # [BLOCK_K]

        # Load x tile: [BLOCK_M, BLOCK_K]
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # Load w tile (weight stored as [N, K]): [BLOCK_N, BLOCK_K]
        w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
        w_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # acc += x @ w^T  →  [BLOCK_M, BLOCK_N]
        acc += tl.dot(x, tl.trans(w))

    # Load scale vector for this column block: [BLOCK_N]
    scale = tl.load(scale_ptr + offs_n, mask=offs_n < N, other=1.0)

    # Fused scale multiply
    acc = acc * scale[None, :]

    # Store output
    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=out_mask)


@torch.fx.wrap
def fused_linear_scale_mul(x, w, scale):
    """
    Fused wrapper: out = (x @ w.T) * scale

    Args:
        x     : input activations  [..., K]
        w     : weight matrix      [N, K]
        scale : channel-wise scale [N]

    Returns:
        out : [..., N]  (same leading dims as x, last dim N)
    """
    leading = x.shape[:-1]
    K = x.shape[-1]
    N = w.shape[0]
    M = x.numel() // K

    x_2d = x.reshape(M, K)
    w_2d = w.reshape(N, K)
    out_2d = torch.empty((M, N), dtype=x.dtype, device=x.device)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),
                         triton.cdiv(N, meta['BLOCK_N']))

    fused_linear_scale_mul_kernel[grid](
        x_2d, w_2d, scale, out_2d,
        M, N, K,
        x_2d.stride(0), x_2d.stride(1),
        w_2d.stride(0), w_2d.stride(1),
    )

    return out_2d.reshape(*leading, N)