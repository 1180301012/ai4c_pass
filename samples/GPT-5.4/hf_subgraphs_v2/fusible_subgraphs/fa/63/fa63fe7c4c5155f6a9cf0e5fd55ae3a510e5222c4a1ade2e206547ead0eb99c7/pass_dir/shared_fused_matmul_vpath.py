import math
import torch
import triton
import triton.language as tl


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.jit
def _smallk_batched_matmul_kernel(
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


@triton.jit
def _slice1_transpose_reshape_split_kernel(
    x_ptr,
    out0_ptr,
    out1_ptr,
    out2_ptr,
    P,
    C,
    S,
    H,
    x_stride0,
    x_stride1,
    x_stride2,
    x_stride3,
    o0_stride0,
    o0_stride1,
    o0_stride2,
    o0_stride3,
    o1_stride0,
    o1_stride1,
    o1_stride2,
    o1_stride3,
    o2_stride0,
    o2_stride1,
    o2_stride2,
    o2_stride3,
    BLOCK_P: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_p = tl.program_id(0)
    pid_bh = tl.program_id(1)

    batch_idx = pid_bh // H
    head_idx = pid_bh % H

    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    offs_c = tl.arange(0, BLOCK_C)

    x_ptrs = (
        x_ptr
        + batch_idx * x_stride0
        + head_idx * x_stride1
        + (offs_p[:, None] + 1) * x_stride2
        + offs_c[None, :] * x_stride3
    )
    valid = (offs_p[:, None] < P) & (offs_c[None, :] < C)
    vals = tl.load(x_ptrs, mask=valid, other=0.0)

    spatial_offsets0 = (
        (offs_p[:, None] // S) * o0_stride2 + (offs_p[:, None] % S) * o0_stride3
    )
    spatial_offsets1 = (
        (offs_p[:, None] // S) * o1_stride2 + (offs_p[:, None] % S) * o1_stride3
    )
    spatial_offsets2 = (
        (offs_p[:, None] // S) * o2_stride2 + (offs_p[:, None] % S) * o2_stride3
    )

    local_h1 = tl.where(head_idx >= 2, head_idx - 2, 0)
    local_h2 = tl.where(head_idx >= 5, head_idx - 5, 0)

    out0_ptrs = (
        out0_ptr
        + batch_idx * o0_stride0
        + (head_idx * C + offs_c[None, :]) * o0_stride1
        + spatial_offsets0
    )
    out1_ptrs = (
        out1_ptr
        + batch_idx * o1_stride0
        + (local_h1 * C + offs_c[None, :]) * o1_stride1
        + spatial_offsets1
    )
    out2_ptrs = (
        out2_ptr
        + batch_idx * o2_stride0
        + (local_h2 * C + offs_c[None, :]) * o2_stride1
        + spatial_offsets2
    )

    tl.store(out0_ptrs, vals, mask=valid & (head_idx < 2))
    tl.store(out1_ptrs, vals, mask=valid & (head_idx >= 2) & (head_idx < 5))
    tl.store(out2_ptrs, vals, mask=valid & (head_idx >= 5))


@torch.fx.wrap
def fused_matmul_vpath(in_0, in_1, in_2):
    batch = in_1.shape[0]
    heads = in_1.shape[1]
    m = in_1.shape[2]
    k = in_1.shape[3]
    n = in_0.shape[3]

    out_matmul = torch.empty((batch, heads, m, n), device=in_1.device, dtype=in_1.dtype)

    max_kn = n if n >= k else k
    if max_kn <= 16:
        block_m = 128
        block_n = 16
        block_k = 16
        num_warps = 4
    elif max_kn <= 32:
        block_m = 64
        block_n = 32
        block_k = 32
        num_warps = 4
    else:
        block_m = 32
        block_n = 64
        block_k = 64
        num_warps = 8

    grid = (triton.cdiv(m, block_m), batch * heads)
    stride_in1 = in_1.stride()
    stride_in0 = in_0.stride()
    stride_out = out_matmul.stride()
    _smallk_batched_matmul_kernel[grid](
        in_1,
        in_0,
        out_matmul,
        m,
        n,
        k,
        heads,
        stride_in1[0],
        stride_in1[1],
        stride_in1[2],
        stride_in1[3],
        stride_in0[0],
        stride_in0[1],
        stride_in0[2],
        stride_in0[3],
        stride_out[0],
        stride_out[1],
        stride_out[2],
        stride_out[3],
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=2,
    )

    p = in_2.shape[2] - 1
    c = in_2.shape[3]
    s = math.isqrt(p)

    out0 = torch.empty((batch, 2 * c, s, s), device=in_2.device, dtype=in_2.dtype)
    out1 = torch.empty((batch, 3 * c, s, s), device=in_2.device, dtype=in_2.dtype)
    out2 = torch.empty((batch, 3 * c, s, s), device=in_2.device, dtype=in_2.dtype)

    if c <= 16:
        block_c = 16
        block_p = 128 if p >= 512 else 64
        num_warps_v = 4
    elif c <= 32:
        block_c = 32
        block_p = 128 if p >= 512 else 64
        num_warps_v = 4
    else:
        block_c = 64
        block_p = 128
        num_warps_v = 8

    grid_v = (triton.cdiv(p, block_p), batch * heads)
    stride_in2 = in_2.stride()
    stride_o0 = out0.stride()
    stride_o1 = out1.stride()
    stride_o2 = out2.stride()
    _slice1_transpose_reshape_split_kernel[grid_v](
        in_2,
        out0,
        out1,
        out2,
        p,
        c,
        s,
        heads,
        stride_in2[0],
        stride_in2[1],
        stride_in2[2],
        stride_in2[3],
        stride_o0[0],
        stride_o0[1],
        stride_o0[2],
        stride_o0[3],
        stride_o1[0],
        stride_o1[1],
        stride_o1[2],
        stride_o1[3],
        stride_o2[0],
        stride_o2[1],
        stride_o2[2],
        stride_o2[3],
        BLOCK_P=block_p,
        BLOCK_C=block_c,
        num_warps=num_warps_v,
        num_stages=2,
    )

    return out_matmul, out0, out1, out2


def replacement_func():
    return fused_matmul_vpath