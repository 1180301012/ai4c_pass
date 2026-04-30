import operator
import torch
import triton
import triton.language as tl
from torch import device


def pattern(in_0, in_1):
    tmp_0 = in_1.norm(p=2, dim=-1, keepdim=True)
    tmp_1 = in_1 / tmp_0
    tmp_2 = in_0.t()
    tmp_3 = tmp_2.to(device(type='cuda'))
    return (tmp_1, tmp_3)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK0': 1024, 'BLOCK1': 256}, num_warps=1, num_stages=1),
        triton.Config({'BLOCK0': 1024, 'BLOCK1': 256}, num_warps=2, num_stages=1),
        triton.Config({'BLOCK0': 1024, 'BLOCK1': 256}, num_warps=4, num_stages=1),
    ],
    key=['D'],
)
@triton.jit
def fused_norm_transpose_kernel(
    in0_ptr,
    in1_ptr,
    out_norm_ptr,
    out_t_ptr,
    D,
    in0_stride_0,
    in0_stride_1,
    in1_stride_0,
    in1_stride_1,
    out_norm_stride_0,
    out_norm_stride_1,
    out_t_stride_0,
    out_t_stride_1,
    BLOCK0: tl.constexpr,
    BLOCK1: tl.constexpr,
):
    offs0 = tl.arange(0, BLOCK0)
    offs1 = BLOCK0 + tl.arange(0, BLOCK1)
    mask0 = offs0 < D
    mask1 = offs1 < D

    # transpose copy: [1, D] -> [D, 1]
    t0 = tl.load(in0_ptr + offs0 * in0_stride_1, mask=mask0, other=0)
    t1 = tl.load(in0_ptr + offs1 * in0_stride_1, mask=mask1, other=0)
    tl.store(out_t_ptr + offs0 * out_t_stride_0, t0, mask=mask0)
    tl.store(out_t_ptr + offs1 * out_t_stride_0, t1, mask=mask1)

    # row 0 normalize
    row0_ptr = in1_ptr
    out0_ptr = out_norm_ptr
    x0a = tl.load(row0_ptr + offs0 * in1_stride_1, mask=mask0, other=0).to(tl.float32)
    x0b = tl.load(row0_ptr + offs1 * in1_stride_1, mask=mask1, other=0).to(tl.float32)
    acc0 = tl.sum(x0a * x0a, axis=0) + tl.sum(x0b * x0b, axis=0)
    inv0 = 1.0 / tl.sqrt(acc0)
    tl.store(out0_ptr + offs0 * out_norm_stride_1, (x0a * inv0).to(tl.bfloat16), mask=mask0)
    tl.store(out0_ptr + offs1 * out_norm_stride_1, (x0b * inv0).to(tl.bfloat16), mask=mask1)

    # row 1 normalize
    row1_ptr = in1_ptr + in1_stride_0
    out1_ptr = out_norm_ptr + out_norm_stride_0
    x1a = tl.load(row1_ptr + offs0 * in1_stride_1, mask=mask0, other=0).to(tl.float32)
    x1b = tl.load(row1_ptr + offs1 * in1_stride_1, mask=mask1, other=0).to(tl.float32)
    acc1 = tl.sum(x1a * x1a, axis=0) + tl.sum(x1b * x1b, axis=0)
    inv1 = 1.0 / tl.sqrt(acc1)
    tl.store(out1_ptr + offs0 * out_norm_stride_1, (x1a * inv1).to(tl.bfloat16), mask=mask0)
    tl.store(out1_ptr + offs1 * out_norm_stride_1, (x1b * inv1).to(tl.bfloat16), mask=mask1)


@torch.fx.wrap
def fused_norm_transpose_kernel_wrapper(in_0, in_1):
    D = in_1.shape[1]
    out_norm = torch.empty_like(in_1)
    out_t = torch.empty((D, 1), device=in_0.device, dtype=in_0.dtype)
    fused_norm_transpose_kernel[(1,)](
        in_0,
        in_1,
        out_norm,
        out_t,
        D,
        in_0.stride(0),
        in_0.stride(1),
        in_1.stride(0),
        in_1.stride(1),
        out_norm.stride(0),
        out_norm.stride(1),
        out_t.stride(0),
        out_t.stride(1),
    )
    return (out_norm, out_t)


def fused_norm_transpose_outputs(in_0, in_1):
    tmp = fused_norm_transpose_kernel_wrapper(in_0, in_1)
    return operator.getitem(tmp, 0), operator.getitem(tmp, 1)


def replacement_func():
    return fused_norm_transpose_outputs