import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _fused_linear_view_transpose_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    M,
    N,
    S,
    K,
    stride_x0,
    stride_x1,
    stride_x2,
    stride_w0,
    stride_w1,
    stride_o0,
    stride_o1,
    stride_o2,
    stride_o3,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    b_idx = offs_m // S
    s_idx = offs_m % S

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        x_ptrs = (
            x_ptr
            + b_idx[:, None] * stride_x0
            + s_idx[:, None] * stride_x1
            + (k0 + offs_k)[None, :] * stride_x2
        )
        x = tl.load(
            x_ptrs,
            mask=(offs_m[:, None] < M) & ((k0 + offs_k)[None, :] < K),
            other=0.0,
        )

        w_ptrs = (
            w_ptr
            + offs_n[None, :] * stride_w0
            + (k0 + offs_k)[:, None] * stride_w1
        )
        w = tl.load(
            w_ptrs,
            mask=(offs_n[None, :] < N) & ((k0 + offs_k)[:, None] < K),
            other=0.0,
        )
        acc += tl.dot(x, w)

    bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc += bias[None, :]

    head_idx = offs_n // 64
    dim_idx = offs_n % 64
    out_ptrs = (
        out_ptr
        + b_idx[:, None] * stride_o0
        + head_idx[None, :] * stride_o1
        + s_idx[:, None] * stride_o2
        + dim_idx[None, :] * stride_o3
    )
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=mask)


@torch.fx.wrap
def fused_linear_view_transpose(x, w, b):
    B = x.shape[0]
    S = x.shape[1]
    K = x.shape[2]
    N = w.shape[0]
    H = N // 64

    if x.dtype == torch.float32 or (B * S) < 256:
        y = F.linear(x, w, b)
        return y.view(B, S, H, 64).transpose(1, 2).contiguous()

    out = torch.empty((B, H, S, 64), device=x.device, dtype=x.dtype)
    M = B * S

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))
    _fused_linear_view_transpose_kernel[grid](
        x,
        w,
        b,
        out,
        M,
        N,
        S,
        K,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        w.stride(0),
        w.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
    )
    return out


def replacement_args_shared(in_0, in_1, in_3):
    return (in_3, in_1, in_0)


def replacement_func_shared():
    return fused_linear_view_transpose