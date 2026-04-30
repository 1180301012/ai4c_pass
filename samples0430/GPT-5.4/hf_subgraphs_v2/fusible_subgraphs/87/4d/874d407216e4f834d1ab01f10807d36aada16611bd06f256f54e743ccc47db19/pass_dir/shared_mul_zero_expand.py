import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 16}, num_warps=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 16}, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 16}, num_warps=4),
    ],
    key=['M', 'N', 'ZERO_M', 'ZERO_N'],
)
@triton.jit
def mul_and_zero_kernel(
    weight_ptr,
    x_ptr,
    out_mul_ptr,
    out_zero_ptr,
    M,
    N,
    ZERO_M,
    ZERO_N,
    stride_xm,
    stride_xn,
    stride_om,
    stride_on,
    stride_zm,
    stride_zn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    offs_m = pid0 * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid1 * BLOCK_N + tl.arange(0, BLOCK_N)

    weight = tl.load(weight_ptr + offs_m, mask=offs_m < M, other=0.0)
    x = tl.load(
        x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        other=0.0,
    )
    y = x * weight[:, None]
    tl.store(
        out_mul_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        y,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )

    zmask = (offs_m[:, None] < ZERO_M) & (offs_n[None, :] < ZERO_N)
    tl.store(
        out_zero_ptr + offs_m[:, None] * stride_zm + offs_n[None, :] * stride_zn,
        0.0,
        mask=zmask,
    )


@torch.fx.wrap
def fused_mul_zero_expand(in_0, in_1, in_2, zero_rows, zero_cols):
    m = in_1.shape[0]
    n = in_2.shape[1]

    out_mul = torch.empty_like(in_2)
    out_zero = torch.empty((zero_rows, zero_cols), device=in_2.device, dtype=in_2.dtype)

    grid = lambda META: (
        triton.cdiv(max(m, zero_rows), META['BLOCK_M']),
        triton.cdiv(max(n, zero_cols), META['BLOCK_N']),
    )

    mul_and_zero_kernel[grid](
        in_1,
        in_2,
        out_mul,
        out_zero,
        m,
        n,
        zero_rows,
        zero_cols,
        in_2.stride(0),
        in_2.stride(1),
        out_mul.stride(0),
        out_mul.stride(1),
        out_zero.stride(0),
        out_zero.stride(1),
    )

    out_expand = in_0.view((-1, 1)).expand_as(out_mul)
    return out_expand, out_zero, out_mul