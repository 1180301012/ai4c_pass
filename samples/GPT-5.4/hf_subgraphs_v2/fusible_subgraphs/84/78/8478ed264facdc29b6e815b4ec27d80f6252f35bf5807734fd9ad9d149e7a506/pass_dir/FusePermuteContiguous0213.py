import torch
import triton
import triton.language as tl


@triton.jit
def _permute0213_contig_kernel(
    x_ptr,
    out_ptr,
    stride_x_b, stride_x_h, stride_x_m, stride_x_d,
    stride_o_b, stride_o_m, stride_o_h, stride_o_d,
    H, M, D,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    m_mask = offs_m < M
    d_mask = offs_d < D
    mask = m_mask[:, None] & d_mask[None, :]

    x_ptrs = (
        x_ptr
        + b * stride_x_b
        + h * stride_x_h
        + offs_m[:, None] * stride_x_m
        + offs_d[None, :] * stride_x_d
    )
    vals = tl.load(x_ptrs, mask=mask, other=0.0)

    o_ptrs = (
        out_ptr
        + b * stride_o_b
        + offs_m[:, None] * stride_o_m
        + h * stride_o_h
        + offs_d[None, :] * stride_o_d
    )
    tl.store(o_ptrs, vals, mask=mask)


@torch.fx.wrap
def permute0213_contiguous(x):
    B = int(x.shape[0])
    H = int(x.shape[1])
    M = int(x.shape[2])
    D = int(x.shape[3])

    out = torch.empty((B, M, H, D), device=x.device, dtype=x.dtype)

    block_m = 64 if M >= 1024 else 32
    block_d = 64 if D > 32 else 32
    num_warps = 4 if block_d <= 32 else 8

    grid = (triton.cdiv(M, block_m), B * H)
    _permute0213_contig_kernel[grid](
        x,
        out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        H, M, D,
        BLOCK_M=block_m,
        BLOCK_D=block_d,
        num_warps=num_warps,
        num_stages=2,
    )
    return out


def pattern(x):
    y = x.permute(0, 2, 1, 3)
    z = y.contiguous()
    return z



def replacement_args(x):
    return (x,)



def replacement_func():
    return permute0213_contiguous