import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['TOTAL'],
)
@triton.jit
def edge_weight_mul_flat16_kernel(
    weight_ptr,
    x_ptr,
    out_ptr,
    TOTAL,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < TOTAL
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    row = offs >> 4
    w = tl.load(weight_ptr + row, mask=mask, other=0.0)
    tl.store(out_ptr + offs, x * w, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['TOTAL'],
)
@triton.jit
def edge_weight_mul_flat128_kernel(
    weight_ptr,
    x_ptr,
    out_ptr,
    TOTAL,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < TOTAL
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    row = offs >> 7
    w = tl.load(weight_ptr + row, mask=mask, other=0.0)
    tl.store(out_ptr + offs, x * w, mask=mask)


@triton.jit
def edge_weight_mul_generic_kernel(
    weight_ptr,
    x_ptr,
    out_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    w = tl.load(weight_ptr + offs_m, mask=offs_m < M, other=0.0)
    x = tl.load(
        x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn,
        mask=mask,
        other=0.0,
    )
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        x * w[:, None],
        mask=mask,
    )


@torch.fx.wrap
def fused_edge_weight_mul(in_1, in_2):
    m = in_1.shape[0]
    n = in_2.shape[1]
    out = torch.empty_like(in_2)
    total = in_2.numel()

    if n == 16:
        grid = lambda META: (triton.cdiv(total, META['BLOCK_SIZE']),)
        edge_weight_mul_flat16_kernel[grid](in_1, in_2, out, total)
    elif n == 128:
        grid = lambda META: (triton.cdiv(total, META['BLOCK_SIZE']),)
        edge_weight_mul_flat128_kernel[grid](in_1, in_2, out, total)
    else:
        grid = lambda META: (
            triton.cdiv(m, META['BLOCK_M']),
            triton.cdiv(n, META['BLOCK_N']),
        )
        edge_weight_mul_generic_kernel[grid](
            in_1,
            in_2,
            out,
            m,
            n,
            in_2.stride(0),
            in_2.stride(1),
            out.stride(0),
            out.stride(1),
            BLOCK_M=64,
            BLOCK_N=64,
        )
    return out