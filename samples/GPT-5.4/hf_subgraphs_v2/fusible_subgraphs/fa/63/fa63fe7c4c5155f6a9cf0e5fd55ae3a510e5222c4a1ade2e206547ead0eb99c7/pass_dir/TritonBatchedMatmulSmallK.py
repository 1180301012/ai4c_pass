import torch
import triton
import triton.language as tl



def pattern(in_0, in_1):
    return in_1 @ in_0



def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def _batched_smallk_matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    H,
    stride_a0,
    stride_a1,
    stride_a2,
    stride_a3,
    stride_b0,
    stride_b1,
    stride_b2,
    stride_b3,
    stride_c0,
    stride_c1,
    stride_c2,
    stride_c3,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    batch_idx = pid_bh // H
    head_idx = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = (
        a_ptr
        + batch_idx * stride_a0
        + head_idx * stride_a1
        + offs_m[:, None] * stride_a2
        + offs_k[None, :] * stride_a3
    )
    b_ptrs = (
        b_ptr
        + batch_idx * stride_b0
        + head_idx * stride_b1
        + offs_k[:, None] * stride_b2
        + offs_n[None, :] * stride_b3
    )

    a = tl.load(
        a_ptrs,
        mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
        other=0.0,
    )
    b = tl.load(
        b_ptrs,
        mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
        other=0.0,
    )

    acc = tl.dot(a, b)

    c_ptrs = (
        c_ptr
        + batch_idx * stride_c0
        + head_idx * stride_c1
        + offs_m[:, None] * stride_c2
        + offs_n[None, :] * stride_c3
    )
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@torch.fx.wrap
def triton_batched_smallk_matmul(in_0, in_1):
    batch = in_1.shape[0]
    heads = in_1.shape[1]
    m = in_1.shape[2]
    k = in_1.shape[3]
    n = in_0.shape[3]

    out = torch.empty((batch, heads, m, n), device=in_1.device, dtype=in_1.dtype)

    max_kn = n if n >= k else k
    if max_kn <= 16:
        block_m = 128
        block_n = 16
        block_k = 16
        num_warps = 4
    elif max_kn <= 32:
        block_m = 128
        block_n = 32
        block_k = 32
        num_warps = 4
    elif max_kn <= 40:
        block_m = 128
        block_n = 64
        block_k = 64
        num_warps = 4
    else:
        block_m = 64
        block_n = 64
        block_k = 64
        num_warps = 8

    grid = (triton.cdiv(m, block_m), batch * heads)
    stride_a = in_1.stride()
    stride_b = in_0.stride()
    stride_c = out.stride()

    _batched_smallk_matmul_kernel[grid](
        in_1,
        in_0,
        out,
        m,
        n,
        k,
        heads,
        stride_a[0],
        stride_a[1],
        stride_a[2],
        stride_a[3],
        stride_b[0],
        stride_b[1],
        stride_b[2],
        stride_b[3],
        stride_c[0],
        stride_c[1],
        stride_c[2],
        stride_c[3],
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=2,
    )
    return out



def replacement_func():
    return triton_batched_smallk_matmul