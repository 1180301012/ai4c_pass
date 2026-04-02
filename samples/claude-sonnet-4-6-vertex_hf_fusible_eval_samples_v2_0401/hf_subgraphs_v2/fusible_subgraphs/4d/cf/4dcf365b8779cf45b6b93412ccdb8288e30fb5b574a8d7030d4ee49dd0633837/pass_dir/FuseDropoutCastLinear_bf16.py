"""
Fuse: x.to(bfloat16) + linear(cast, weight, bias)
Matches RECT_L bfloat16 graph (M=128, N=128, K=128).
When dropout has p=0.0, FX removes it leaving only: to(bf16) -> linear.
"""
import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    to = in_2.to(torch.bfloat16)
    linear = torch.nn.functional.linear(to, in_1, in_0)
    return linear


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32,  'BLOCK_K': 32, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64,  'BLOCK_K': 32, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32,  'BLOCK_K': 32, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_K': 32, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32,  'BLOCK_K': 32, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64,  'BLOCK_K': 32, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32,  'BLOCK_K': 64, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64,  'BLOCK_K': 64, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32,  'BLOCK_K': 64, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_K': 64, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32,  'BLOCK_K': 64, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64,  'BLOCK_K': 64, 'num_stages': 2, 'num_warps': 4}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_cast_linear_bf16_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    out[M,N] = x[M,K] @ w[N,K].T + b[N]   (all bfloat16)
    Loads w as [BLOCK_N, BLOCK_K] (coalesced in K), uses tl.trans for dot.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k_start * BLOCK_K + tl.arange(0, BLOCK_K)
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
        w_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        acc = tl.dot(x, tl.trans(w), acc, out_dtype=tl.float32)

    b = tl.load(b_ptr + offs_n, mask=(offs_n < N), other=0.0).to(tl.float32)
    acc += b[None, :]
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=out_mask)


@torch.fx.wrap
def dropout_cast_linear_bf16(bias, weight, x):
    """
    Drop-in for dropout(p=0,training=False) + .to(bf16) + linear.
    No blocked APIs: only torch.empty, .reshape(), .contiguous(), .stride(), Triton kernel.
    x/weight/bias are already bfloat16 at inference time.
    """
    orig_shape = x.shape
    K = weight.shape[1]
    N = weight.shape[0]
    x_2d = x.reshape(-1, K).contiguous()
    weight_c = weight.contiguous()
    bias_c = bias.contiguous()
    M = x_2d.shape[0]
    out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    fused_cast_linear_bf16_kernel[grid](
        x_2d, weight_c, bias_c, out,
        M, N, K,
        x_2d.stride(0), x_2d.stride(1),
        weight_c.stride(0), weight_c.stride(1),
        out.stride(0), out.stride(1),
    )
    return out.reshape(*orig_shape[:-1], N)


def replacement_func():
    return dropout_cast_linear_bf16