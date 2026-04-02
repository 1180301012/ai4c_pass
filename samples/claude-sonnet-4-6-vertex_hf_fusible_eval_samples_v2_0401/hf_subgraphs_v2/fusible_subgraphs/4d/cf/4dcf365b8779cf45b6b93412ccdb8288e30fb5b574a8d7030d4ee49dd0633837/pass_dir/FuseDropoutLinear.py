"""
Fuse: dropout(x, 0.1, False, False) + linear(dropout_out, weight, bias)
Matches both bigbird bfloat16 and float16 graphs (same ops, different tensor dtypes).
Dropout with training=False is identity, so we skip it and run a fused Triton GEMM+bias.

Key optimization: load weight tile as [BLOCK_N, BLOCK_K] (coalesced in K) then
use tl.trans() so tl.dot sees the correct [BLOCK_K, BLOCK_N] transposed tile.
"""
import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    tmp_3 = torch.nn.functional.dropout(in_2, 0.1, False, False)
    linear = torch.nn.functional.linear(tmp_3, in_1, in_0)
    return linear


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.autotune(
    configs=[
        # Small-M configs for bigbird M=17, N=3072, K=768
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32,  'BLOCK_K': 32, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64,  'BLOCK_K': 32, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 32, 'num_stages': 2, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32,  'BLOCK_K': 64, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64,  'BLOCK_K': 64, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64, 'num_stages': 2, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32,  'BLOCK_K': 32, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_K': 32, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 32, 'num_stages': 2, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_K': 64, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64, 'num_stages': 2, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64,  'BLOCK_K': 32, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64,  'BLOCK_K': 64, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'num_stages': 2, 'num_warps': 8}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_drop_linear_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    IS_BF16: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Computes out[M, N] = x[M, K] @ weight[N, K].T + bias[N]

    Weight loading strategy for memory coalescing:
      Load weight tile as [BLOCK_N, BLOCK_K] (coalesced in K dimension),
      then call tl.trans() to get [BLOCK_K, BLOCK_N] for tl.dot.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k_start * BLOCK_K + tl.arange(0, BLOCK_K)

        # x tile [BLOCK_M, BLOCK_K] — coalesced in K
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # w tile [BLOCK_N, BLOCK_K] — load coalesced in K direction
        # w[n, k] = w_ptr + n * stride_wn + k * stride_wk
        # For fixed n, varying k → stride 1 → COALESCED
        w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
        w_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)  # [BLOCK_N, BLOCK_K]

        # Transpose to [BLOCK_K, BLOCK_N] for tl.dot(x [M,K], w.T [K,N])
        acc = tl.dot(x, tl.trans(w), acc, out_dtype=tl.float32)

    # Bias addition
    b = tl.load(b_ptr + offs_n, mask=(offs_n < N), other=0.0).to(tl.float32)
    acc += b[None, :]

    # Store with dtype cast
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    if IS_BF16:
        tl.store(out_ptrs, acc.to(tl.bfloat16), mask=out_mask)
    else:
        tl.store(out_ptrs, acc.to(tl.float16), mask=out_mask)


@torch.fx.wrap
def dropout_linear_fused(bias, weight, x):
    """
    Drop-in replacement for dropout(training=False) + linear.
    bias:   [N]
    weight: [N, K]
    x:      [..., K]
    returns [..., N]
    """
    orig_shape = x.shape
    K = weight.shape[1]
    N = weight.shape[0]

    x_2d = x.reshape(-1, K).contiguous()
    weight_c = weight.contiguous()
    bias_c = bias.contiguous()
    M = x_2d.shape[0]

    is_bf16 = (x.dtype == torch.bfloat16)
    out = torch.empty((M, N), dtype=x.dtype, device=x.device)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )

    fused_drop_linear_kernel[grid](
        x_2d, weight_c, bias_c, out,
        M, N, K,
        x_2d.stride(0), x_2d.stride(1),
        weight_c.stride(0), weight_c.stride(1),
        out.stride(0), out.stride(1),
        IS_BF16=is_bf16,
    )

    return out.reshape(*orig_shape[:-1], N)


def replacement_func():
    return dropout_linear_fused