import torch
import triton
import triton.language as tl


# Match only the division. The output is consumed by transpose outside the matched subgraph,
# so it must be returned by the pattern.
def pattern(in_0, scale):
    tmp_0 = in_0 / scale
    return tmp_0


def replacement_args(in_0, scale):
    return (in_0, scale)


@triton.jit
def _scale_contiguous_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    scale_inv,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = x * scale_inv
    tl.store(out_ptr + offsets, y, mask=mask)


@triton.jit
def _scale_strided_4d_kernel(
    x_ptr,
    out_ptr,
    x_stride_0,
    x_stride_1,
    x_stride_2,
    x_stride_3,
    out_stride_0,
    out_stride_1,
    out_stride_2,
    out_stride_3,
    H,
    M,
    K,
    scale_inv,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_k = tl.program_id(2)

    b = pid_bh // H
    h = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)

    x_ptrs = (
        x_ptr
        + b * x_stride_0
        + h * x_stride_1
        + offs_m[:, None] * x_stride_2
        + offs_k[None, :] * x_stride_3
    )
    y = tl.load(x_ptrs, mask=mask, other=0.0) * scale_inv

    out_ptrs = (
        out_ptr
        + b * out_stride_0
        + h * out_stride_1
        + offs_m[:, None] * out_stride_2
        + offs_k[None, :] * out_stride_3
    )
    tl.store(out_ptrs, y, mask=mask)


@torch.fx.wrap
def scale_only(in_0, scale):
    scale_inv = 1.0 / scale
    out = torch.empty_like(in_0)

    if in_0.is_contiguous():
        n_elements = in_0.numel()
        if n_elements <= 4096:
            block_size = 256
            num_warps = 1
        elif n_elements <= 65536:
            block_size = 1024
            num_warps = 2
        else:
            block_size = 2048
            num_warps = 4

        grid = (triton.cdiv(n_elements, block_size),)
        _scale_contiguous_kernel[grid](
            in_0,
            out,
            n_elements,
            scale_inv,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            num_stages=1,
        )
        return out

    B = in_0.shape[0]
    H = in_0.shape[1]
    M = in_0.shape[2]
    K = in_0.shape[3]

    if K <= 8:
        block_k = 8
        num_warps = 1 if M <= 16 else 2
        block_m = 16 if M <= 16 else (32 if M <= 64 else 64)
    else:
        block_k = 32
        num_warps = 2 if M <= 32 else 4
        block_m = 32 if M <= 32 else 64

    grid = (B * H, triton.cdiv(M, block_m), triton.cdiv(K, block_k))
    _scale_strided_4d_kernel[grid](
        in_0,
        out,
        in_0.stride(0),
        in_0.stride(1),
        in_0.stride(2),
        in_0.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        H,
        M,
        K,
        scale_inv,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=1,
    )
    return out


def replacement_func():
    return scale_only