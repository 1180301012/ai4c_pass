import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────────────────────
# Kernel: GEMM with bias
#   out[m, n] = sum_k( x[m, k] * w[n, k] ) + b[n]
#   x : [M, K]  (input)
#   w : [N, K]  (weight)
#   b : [N]     (bias)
#   out: [M, N]
#
#   Tile-wise GEMM using tl.dot (fp32 accumulation regardless of input dtype)
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64},  num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64,  'BLOCK_K': 32},  num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_K': 32},  num_warps=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 32},  num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_bias_kernel(
    x_ptr,     # input   [M, K]
    w_ptr,     # weight  [N, K]
    b_ptr,     # bias    [N]
    o_ptr,     # output  [M, N]
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        ).to(tl.float32)

        w = tl.load(
            w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
            mask=(offs_n[:, None] < N) & (offs_k[None, :] < K),
            other=0.0,
        ).to(tl.float32)

        acc += tl.dot(x, tl.trans(w))

    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc += b[None, :]

    tl.store(
        o_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper
# ─────────────────────────────────────────────────────────────────────────────
@torch.fx.wrap
def linear_bias_torch(in_0, in_1, in_2):
    """
    in_0 : bias   [2]
    in_1 : weight [2, 448]
    in_2 : input  [B, 448]
    Returns : [B, 2]
    """
    B = in_2.shape[0]
    K = in_2.shape[1]
    M = in_1.shape[0]   # 2
    N = in_1.shape[1]   # 448

    out = torch.empty((B, M), dtype=in_2.dtype, device=in_2.device)

    grid = lambda meta: (
        triton.cdiv(B, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )
    linear_bias_kernel[grid](
        in_2, in_1, in_0, out,
        B, M, K,
        in_2.stride(0), in_2.stride(1),
        in_1.stride(0), in_1.stride(1),
        out.stride(0), out.stride(1),
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Pattern / replacement_args / replacement_func
# ─────────────────────────────────────────────────────────────────────────────
def pattern(in_0, in_1, in_2):
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    return linear


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return linear_bias_torch