import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = linear.transpose(-1, -2)
    tmp_4 = in_3 * tmp_3
    return tmp_4


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128, 'BLOCK_M': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_N': 64, 'BLOCK_M': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_N': 128, 'BLOCK_M': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_N': 64, 'BLOCK_M': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_N': 128, 'BLOCK_M': 32, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
    ],
    key=['N', 'M', 'K'],
)
@triton.jit
def _fused_linear_transpose_mul_kernel(
    bias_ptr,
    weight_ptr,
    x_ptr,
    u_ptr,
    out_ptr,
    bias_s0,
    w_s0,
    w_s1,
    x_s0,
    x_s1,
    x_s2,
    u_s0,
    u_s1,
    u_s2,
    out_s0,
    out_s1,
    out_s2,
    N,
    M,
    K,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        curr_k = k_start + offs_k

        x_ptrs = x_ptr + pid_b * x_s0 + offs_n[:, None] * x_s1 + curr_k[None, :] * x_s2
        w_ptrs = weight_ptr + offs_m[None, :] * w_s0 + curr_k[:, None] * w_s1

        x_mask = (offs_n[:, None] < N) & (curr_k[None, :] < K)
        w_mask = (offs_m[None, :] < M) & (curr_k[:, None] < K)

        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        acc += tl.dot(x, w)

    bias = tl.load(bias_ptr + offs_m * bias_s0, mask=offs_m < M, other=0.0)
    acc += bias[None, :]

    out_mask = (offs_n[:, None] < N) & (offs_m[None, :] < M)
    u_ptrs = u_ptr + pid_b * u_s0 + offs_m[None, :] * u_s1 + offs_n[:, None] * u_s2
    u = tl.load(u_ptrs, mask=out_mask, other=0.0)

    out = acc * u
    out_ptrs = out_ptr + pid_b * out_s0 + offs_m[None, :] * out_s1 + offs_n[:, None] * out_s2
    tl.store(out_ptrs, out, mask=out_mask)


@torch.fx.wrap
def fused_linear_transpose_mul(in_0, in_1, in_2, in_3):
    b = in_2.shape[0]
    n = in_2.shape[1]
    k = in_2.shape[2]
    m = in_1.shape[0]

    out = torch.empty_like(in_3)

    grid = lambda META: (
        triton.cdiv(n, META['BLOCK_N']),
        triton.cdiv(m, META['BLOCK_M']),
        b,
    )

    _fused_linear_transpose_mul_kernel[grid](
        in_0,
        in_1,
        in_2,
        in_3,
        out,
        in_0.stride(0),
        in_1.stride(0),
        in_1.stride(1),
        in_2.stride(0),
        in_2.stride(1),
        in_2.stride(2),
        in_3.stride(0),
        in_3.stride(1),
        in_3.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        n,
        m,
        k,
    )

    return out


def replacement_func():
    return fused_linear_transpose_mul