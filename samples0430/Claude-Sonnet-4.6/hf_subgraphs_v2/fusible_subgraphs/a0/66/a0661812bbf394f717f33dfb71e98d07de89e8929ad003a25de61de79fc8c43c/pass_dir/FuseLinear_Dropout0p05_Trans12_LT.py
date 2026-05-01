import torch
from pass_dir.kernel_impl import fused_linear as _dispatch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32,  'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 16,  'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 16,  'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 16,  'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 16,  'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 16,  'BLOCK_K': 16, 'num_stages': 3, 'num_warps': 2}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 16,  'BLOCK_K': 16, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32,  'BLOCK_K': 16, 'num_stages': 3, 'num_warps': 4}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _linear_bias_fwd_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    DTYPE: tl.constexpr,
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

        x_tile = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )

        w_tile = tl.load(
            w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn,
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )

        acc += tl.dot(x_tile, w_tile)

    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + b[None, :].to(tl.float32)

    if DTYPE == 1:
        out = acc.to(tl.float16)
    elif DTYPE == 2:
        out = acc.to(tl.bfloat16)
    else:
        out = acc.to(tl.float32)

    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        out,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@torch.fx.wrap
def _linear_bias_transpose_dispatch(in_0, in_1, in_2, route):
    B = in_2.shape[0]
    N_seq = in_2.shape[1]
    K = in_2.shape[2]
    N_out = in_1.shape[0]
    M = B * N_seq

    x_2d = in_2.view(M, K)
    out_2d = torch.empty((M, N_out), dtype=in_2.dtype, device=in_2.device)

    if in_2.dtype == torch.float16:
        dtype_id = 1
    elif in_2.dtype == torch.bfloat16:
        dtype_id = 2
    else:
        dtype_id = 0

    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N_out, meta['BLOCK_N']))

    _linear_bias_fwd_kernel[grid](
        x_2d, in_1, in_0, out_2d,
        M, N_out, K,
        x_2d.stride(0), x_2d.stride(1),
        in_1.stride(0), in_1.stride(1),
        out_2d.stride(0), out_2d.stride(1),
        DTYPE=dtype_id,
    )

    out_3d = out_2d.view(B, N_seq, N_out)
    t = out_3d.transpose(1, 2)

    if route == "LT":
        return out_3d, t
    else:  # "TL"
        return t, out_3d


# ── pattern: dropout=0.05 (transpose stays in graph) ─────────────────────────
def pattern(in_0, in_1, in_2):
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = torch.nn.functional.dropout(linear, 0.05, False, False)
    return tmp_3


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return _dispatch