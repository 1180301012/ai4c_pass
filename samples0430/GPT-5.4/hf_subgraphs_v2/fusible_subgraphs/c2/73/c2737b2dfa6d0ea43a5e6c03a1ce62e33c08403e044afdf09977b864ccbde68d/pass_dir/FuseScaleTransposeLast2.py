import torch
import triton
import triton.language as tl


# Pattern matching function
# Match: tmp_0 = in_0 / scale; tmp_1 = tmp_0.transpose(-1, -2)
def pattern(in_0, scale):
    tmp_0 = in_0 / scale
    tmp_1 = tmp_0.transpose(-1, -2)
    return tmp_1


# Extract both the tensor and the matched scalar literal.
def replacement_args(in_0, scale):
    return (in_0, scale)


@triton.jit
def _scale_transpose_k8_kernel(
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
    scale,
    BLOCK_M: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offs_m < M

    x_base = x_ptr + b * x_stride_0 + h * x_stride_1 + offs_m * x_stride_2
    out_base = out_ptr + b * out_stride_0 + h * out_stride_1 + offs_m * out_stride_3

    x0 = tl.load(x_base + 0 * x_stride_3, mask=mask, other=0.0) / scale
    x1 = tl.load(x_base + 1 * x_stride_3, mask=mask, other=0.0) / scale
    x2 = tl.load(x_base + 2 * x_stride_3, mask=mask, other=0.0) / scale
    x3 = tl.load(x_base + 3 * x_stride_3, mask=mask, other=0.0) / scale
    x4 = tl.load(x_base + 4 * x_stride_3, mask=mask, other=0.0) / scale
    x5 = tl.load(x_base + 5 * x_stride_3, mask=mask, other=0.0) / scale
    x6 = tl.load(x_base + 6 * x_stride_3, mask=mask, other=0.0) / scale
    x7 = tl.load(x_base + 7 * x_stride_3, mask=mask, other=0.0) / scale

    tl.store(out_base + 0 * out_stride_2, x0, mask=mask)
    tl.store(out_base + 1 * out_stride_2, x1, mask=mask)
    tl.store(out_base + 2 * out_stride_2, x2, mask=mask)
    tl.store(out_base + 3 * out_stride_2, x3, mask=mask)
    tl.store(out_base + 4 * out_stride_2, x4, mask=mask)
    tl.store(out_base + 5 * out_stride_2, x5, mask=mask)
    tl.store(out_base + 6 * out_stride_2, x6, mask=mask)
    tl.store(out_base + 7 * out_stride_2, x7, mask=mask)


@triton.jit
def _scale_transpose_k64_kernel(
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
    scale,
    BLOCK_M: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, 64)
    mask = offs_m[:, None] < M

    x_ptrs = (
        x_ptr
        + b * x_stride_0
        + h * x_stride_1
        + offs_m[:, None] * x_stride_2
        + offs_k[None, :] * x_stride_3
    )
    y = tl.load(x_ptrs, mask=mask, other=0.0) / scale

    out_ptrs = (
        out_ptr
        + b * out_stride_0
        + h * out_stride_1
        + offs_k[:, None] * out_stride_2
        + offs_m[None, :] * out_stride_3
    )
    out_mask = offs_m[None, :] < M
    tl.store(out_ptrs, tl.trans(y), mask=out_mask)


@triton.jit
def _scale_transpose_generic_kernel(
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
    scale,
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
    y = tl.load(x_ptrs, mask=mask, other=0.0) / scale

    out_ptrs = (
        out_ptr
        + b * out_stride_0
        + h * out_stride_1
        + offs_k[:, None] * out_stride_2
        + offs_m[None, :] * out_stride_3
    )
    out_mask = (offs_k[:, None] < K) & (offs_m[None, :] < M)
    tl.store(out_ptrs, tl.trans(y), mask=out_mask)


@torch.fx.wrap
def scale_transpose_last2(in_0, scale):
    B = in_0.shape[0]
    H = in_0.shape[1]
    M = in_0.shape[2]
    K = in_0.shape[3]

    out = torch.empty((B, H, K, M), device=in_0.device, dtype=in_0.dtype)

    if K == 8:
        if M <= 16:
            block_m = 16
            num_warps = 1
        elif M <= 64:
            block_m = 64
            num_warps = 2
        elif M <= 128:
            block_m = 128
            num_warps = 4
        else:
            block_m = 256
            num_warps = 4
        grid = (B * H, triton.cdiv(M, block_m))
        _scale_transpose_k8_kernel[grid](
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
            scale,
            BLOCK_M=block_m,
            num_warps=num_warps,
            num_stages=2,
        )
    elif K == 64:
        block_m = 16 if M <= 16 else 32
        grid = (B * H, triton.cdiv(M, block_m))
        _scale_transpose_k64_kernel[grid](
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
            scale,
            BLOCK_M=block_m,
            num_warps=4,
            num_stages=2,
        )
    else:
        block_k = 32 if K >= 32 else 8
        if M <= 16:
            block_m = 16
            num_warps = 1
        elif M <= 64:
            block_m = 64
            num_warps = 2
        else:
            block_m = 128
            num_warps = 4
        grid = (B * H, triton.cdiv(M, block_m), triton.cdiv(K, block_k))
        _scale_transpose_generic_kernel[grid](
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
            scale,
            BLOCK_M=block_m,
            BLOCK_K=block_k,
            num_warps=num_warps,
            num_stages=2,
        )

    return out


# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return scale_transpose_last2