"""
Shared Triton GEMM kernels + dispatcher used by all passes in this directory.
All pass files import `shared_dispatch` and return it from replacement_func().
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Small BLOCK_K configs
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 32, 'num_stages': 2, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 32, 'num_stages': 2, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64, 'num_stages': 3, 'num_warps': 8}),
        # Large BLOCK_K=128 for K=128 (single loop iteration, minimal overhead)
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64,  'BLOCK_K': 128, 'num_stages': 1, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 128, 'num_stages': 1, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 128, 'num_stages': 1, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_K': 128, 'num_stages': 1, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 128, 'num_stages': 1, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 128, 'num_stages': 1, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64,  'BLOCK_K': 128, 'num_stages': 1, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 128, 'num_stages': 1, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128,'BLOCK_N': 128, 'BLOCK_K': 128, 'num_stages': 1, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64,  'BLOCK_K': 64,  'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128,'BLOCK_N': 128, 'BLOCK_K': 64,  'num_stages': 2, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 128,'BLOCK_N': 256, 'BLOCK_K': 64,  'num_stages': 2, 'num_warps': 8}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _gemm_bias_fp16_kernel(
    a_ptr, b_ptr, c_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """C = A @ B^T + bias, store as float16. A and B are float16."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    a_ptrs = a_ptr + rm[:, None] * stride_am + tl.arange(0, BLOCK_K)[None, :] * stride_ak
    b_ptrs = b_ptr + rn[:, None] * stride_bn + tl.arange(0, BLOCK_K)[None, :] * stride_bk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem = K - k * BLOCK_K
        mask_k = tl.arange(0, BLOCK_K) < k_rem
        a = tl.load(a_ptrs, mask=(rm[:, None] < M) & mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=(rn[:, None] < N) & mask_k[None, :], other=0.0)
        acc += tl.dot(a, tl.trans(b), allow_tf32=False)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias = tl.load(bias_ptr + rn, mask=rn < N, other=0.0).to(tl.float32)
    acc += bias[None, :]

    c = acc.to(tl.float16)
    c_ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 32, 'num_stages': 2, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 32, 'num_stages': 2, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64,  'BLOCK_K': 128, 'num_stages': 1, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 128, 'num_stages': 1, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 128, 'num_stages': 1, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_K': 128, 'num_stages': 1, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 128, 'num_stages': 1, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 128, 'num_stages': 1, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64,  'BLOCK_K': 128, 'num_stages': 1, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 128, 'num_stages': 1, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128,'BLOCK_N': 128, 'BLOCK_K': 128, 'num_stages': 1, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64,  'BLOCK_K': 64,  'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128,'BLOCK_N': 128, 'BLOCK_K': 64,  'num_stages': 2, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 128,'BLOCK_N': 256, 'BLOCK_K': 64,  'num_stages': 2, 'num_warps': 8}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _gemm_bias_bf16_kernel(
    a_ptr, b_ptr, c_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """C = A @ B^T + bias, store as bfloat16. A and B are bfloat16."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    a_ptrs = a_ptr + rm[:, None] * stride_am + tl.arange(0, BLOCK_K)[None, :] * stride_ak
    b_ptrs = b_ptr + rn[:, None] * stride_bn + tl.arange(0, BLOCK_K)[None, :] * stride_bk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem = K - k * BLOCK_K
        mask_k = tl.arange(0, BLOCK_K) < k_rem
        a = tl.load(a_ptrs, mask=(rm[:, None] < M) & mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=(rn[:, None] < N) & mask_k[None, :], other=0.0)
        acc += tl.dot(a, tl.trans(b), allow_tf32=False)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias = tl.load(bias_ptr + rn, mask=rn < N, other=0.0).to(tl.float32)
    acc += bias[None, :]

    c = acc.to(tl.bfloat16)
    c_ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@torch.fx.wrap
def shared_dispatch(bias, weight, x, route):
    """
    Shared replacement function for all dropout+linear fusion passes.
    Uses only whitelisted tensor ops (torch.empty, torch.as_tensor, etc.).
    route="bigbird" -> compute x @ weight.T + bias (no dtype conversion)
    route="rect_l" -> same GEMM, x already in target dtype
    """
    # Pure Python arithmetic for dimensions
    M = x.numel() // x.shape[-1]
    N = weight.shape[0]
    K = weight.shape[1]

    # Allocate output with dtype matching x (not hardcoded route)
    out = torch.empty(
        x.shape[:-1] + (N,) if x.dim() > 1 else (M, N),
        dtype=x.dtype,
        device=x.device,
    )

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

    if x.dtype == torch.bfloat16 and route != "no_gemm":
        _gemm_bias_bf16_kernel[grid](
            x, weight, out, bias,
            M, N, K,
            x.stride(-2) if x.dim() > 1 else x.stride(0),
            x.stride(-1),
            weight.stride(0), weight.stride(1),
            out.stride(0), out.stride(1),
        )
    else:
        _gemm_bias_fp16_kernel[grid](
            x, weight, out, bias,
            M, N, K,
            x.stride(-2) if x.dim() > 1 else x.stride(0),
            x.stride(-1),
            weight.stride(0), weight.stride(1),
            out.stride(0), out.stride(1),
        )

    return out