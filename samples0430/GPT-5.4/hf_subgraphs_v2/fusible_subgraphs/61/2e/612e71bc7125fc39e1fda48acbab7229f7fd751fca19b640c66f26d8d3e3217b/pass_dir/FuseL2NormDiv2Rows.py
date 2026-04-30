import torch
import triton
import triton.language as tl


def pattern(in_1):
    tmp_0 = in_1.norm(p=2, dim=-1, keepdim=True)
    tmp_1 = in_1 / tmp_0
    return tmp_1


def replacement_args(in_1):
    return (in_1,)


@triton.jit
def l2norm_2rows_two_tiles_kernel(
    x_ptr,
    out_ptr,
    D,
    x_stride_0,
    x_stride_1,
    out_stride_0,
    out_stride_1,
    BLOCK0: tl.constexpr,
    BLOCK1: tl.constexpr,
):
    offs0 = tl.arange(0, BLOCK0)
    mask0 = offs0 < D
    offs1 = BLOCK0 + tl.arange(0, BLOCK1)
    mask1 = offs1 < D

    row0_ptr = x_ptr
    row1_ptr = x_ptr + x_stride_0
    out0_ptr = out_ptr
    out1_ptr = out_ptr + out_stride_0

    x00 = tl.load(row0_ptr + offs0 * x_stride_1, mask=mask0, other=0).to(tl.float32)
    x10 = tl.load(row1_ptr + offs0 * x_stride_1, mask=mask0, other=0).to(tl.float32)
    x01 = tl.load(row0_ptr + offs1 * x_stride_1, mask=mask1, other=0).to(tl.float32)
    x11 = tl.load(row1_ptr + offs1 * x_stride_1, mask=mask1, other=0).to(tl.float32)

    acc0 = tl.sum(x00 * x00, axis=0) + tl.sum(x01 * x01, axis=0)
    acc1 = tl.sum(x10 * x10, axis=0) + tl.sum(x11 * x11, axis=0)

    inv0 = 1.0 / tl.sqrt(acc0)
    inv1 = 1.0 / tl.sqrt(acc1)

    tl.store(out0_ptr + offs0 * out_stride_1, (x00 * inv0).to(tl.bfloat16), mask=mask0)
    tl.store(out1_ptr + offs0 * out_stride_1, (x10 * inv1).to(tl.bfloat16), mask=mask0)
    tl.store(out0_ptr + offs1 * out_stride_1, (x01 * inv0).to(tl.bfloat16), mask=mask1)
    tl.store(out1_ptr + offs1 * out_stride_1, (x11 * inv1).to(tl.bfloat16), mask=mask1)


@torch.fx.wrap
def fused_l2norm_2rows(in_1):
    out = torch.empty_like(in_1)
    l2norm_2rows_two_tiles_kernel[(1,)](
        in_1,
        out,
        in_1.shape[1],
        in_1.stride(0),
        in_1.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK0=1024,
        BLOCK1=256,
        num_warps=4,
        num_stages=1,
    )
    return out


def replacement_func():
    return fused_l2norm_2rows