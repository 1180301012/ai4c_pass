import torch
import triton
import triton.language as tl


@triton.jit
def _fused_depthwise_conv_add_permute_kernel(
    weight_ptr,
    ctx_ptr,
    val_ptr,
    out_ptr,
    B,
    H,
    L,
    D,
    w_s0,
    w_s2,
    ctx_s0,
    ctx_s1,
    ctx_s2,
    ctx_s3,
    val_s0,
    val_s1,
    val_s2,
    val_s3,
    out_s0,
    out_s1,
    out_s2,
    out_s3,
    BLOCK_L: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_l = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_bh = tl.program_id(2)

    b = pid_bh // H
    h = pid_bh % H

    offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    d_mask = offs_d[None, :] < D
    out_mask = (offs_l[:, None] < L) & d_mask

    acc = tl.zeros((BLOCK_L, BLOCK_D), dtype=tl.float32)

    for k in range(65):
        src_l = offs_l + k - 32
        src_mask = (src_l[:, None] >= 0) & (src_l[:, None] < L) & d_mask
        val_ptrs = (
            val_ptr
            + b * val_s0
            + h * val_s1
            + src_l[:, None] * val_s2
            + offs_d[None, :] * val_s3
        )
        x = tl.load(val_ptrs, mask=src_mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + h * w_s0 + k * w_s2).to(tl.float32)
        acc += x * w

    ctx_ptrs = (
        ctx_ptr
        + b * ctx_s0
        + h * ctx_s1
        + offs_l[:, None] * ctx_s2
        + offs_d[None, :] * ctx_s3
    )
    ctx = tl.load(ctx_ptrs, mask=out_mask, other=0.0).to(tl.float32)
    out_val = acc + ctx

    tl.store(ctx_ptrs, out_val, mask=out_mask)

    out_ptrs = (
        out_ptr
        + b * out_s0
        + offs_l[:, None] * out_s1
        + h * out_s2
        + offs_d[None, :] * out_s3
    )
    tl.store(out_ptrs, out_val, mask=out_mask)


@triton.jit
def _permute_contiguous_kernel(
    inp_ptr,
    out_ptr,
    B,
    H,
    L,
    D,
    in_s0,
    in_s1,
    in_s2,
    in_s3,
    out_s0,
    out_s1,
    out_s2,
    out_s3,
    BLOCK_L: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_l = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_bh = tl.program_id(2)

    b = pid_bh // H
    h = pid_bh % H

    offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = (offs_l[:, None] < L) & (offs_d[None, :] < D)

    inp_ptrs = (
        inp_ptr
        + b * in_s0
        + h * in_s1
        + offs_l[:, None] * in_s2
        + offs_d[None, :] * in_s3
    )
    x = tl.load(inp_ptrs, mask=mask, other=0.0)

    out_ptrs = (
        out_ptr
        + b * out_s0
        + offs_l[:, None] * out_s1
        + h * out_s2
        + offs_d[None, :] * out_s3
    )
    tl.store(out_ptrs, x, mask=mask)


def _select_config(L, D):
    if D <= 8:
        block_d = 8
        block_l = 64 if L >= 64 else 16
        num_warps = 4
    elif D <= 16:
        block_d = 16
        block_l = 32 if L >= 32 else 16
        num_warps = 4
    elif D <= 32:
        block_d = 32
        block_l = 16
        num_warps = 4
    else:
        block_d = 32
        block_l = 16 if L >= 16 else 8
        num_warps = 8
    return block_l, block_d, num_warps


@torch.fx.wrap
def fused_dispatch(*args):
    route = args[-1]

    if route == "conv_add_permute":
        weight, ctx, val, _ = args
        B, H, L, D = ctx.shape
        out = torch.empty((B, L, H, D), device=ctx.device, dtype=ctx.dtype)
        block_l, block_d, num_warps = _select_config(L, D)
        grid = (triton.cdiv(L, block_l), triton.cdiv(D, block_d), B * H)
        _fused_depthwise_conv_add_permute_kernel[grid](
            weight,
            ctx,
            val,
            out,
            B,
            H,
            L,
            D,
            weight.stride(0),
            weight.stride(2),
            ctx.stride(0),
            ctx.stride(1),
            ctx.stride(2),
            ctx.stride(3),
            val.stride(0),
            val.stride(1),
            val.stride(2),
            val.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            BLOCK_L=block_l,
            BLOCK_D=block_d,
            num_warps=num_warps,
        )
        return out

    if route == "permute_contiguous":
        x, _ = args
        B, H, L, D = x.shape
        out = torch.empty((B, L, H, D), device=x.device, dtype=x.dtype)
        block_l, block_d, num_warps = _select_config(L, D)
        grid = (triton.cdiv(L, block_l), triton.cdiv(D, block_d), B * H)
        _permute_contiguous_kernel[grid](
            x,
            out,
            B,
            H,
            L,
            D,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            x.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            BLOCK_L=block_l,
            BLOCK_D=block_d,
            num_warps=num_warps,
        )
        return out

    raise RuntimeError(f"Unknown fused dispatch route: {route}")


def replacement_func():
    return fused_dispatch