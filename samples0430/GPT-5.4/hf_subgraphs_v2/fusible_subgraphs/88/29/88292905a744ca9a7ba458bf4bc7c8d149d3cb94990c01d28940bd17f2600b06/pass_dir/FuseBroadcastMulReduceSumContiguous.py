import torch
import triton
import triton.language as tl


def pattern(weight, x):
    tmp_0 = weight * x
    tmp_1 = torch.sum(tmp_0, dim=1)
    tmp_2 = tmp_1.contiguous()
    return tmp_2


def replacement_args(weight, x):
    return (weight, x)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_HW": 256}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK_HW": 256}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_HW": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_HW": 512}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK_HW": 512}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_HW": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_HW": 512}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_HW": 1024}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK_HW": 1024}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_HW": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_HW": 2048}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_HW": 2048}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_HW": 4096}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_HW": 4096}, num_warps=8, num_stages=2),
    ],
    key=["HW"],
)
@triton.jit
def _weighted_sum2_kernel(
    weight_ptr,
    x_ptr,
    out_ptr,
    HW,
    w_batch_stride,
    w_group_stride,
    w_channel_stride,
    x_batch_stride,
    x_group_stride,
    x_channel_stride,
    out_batch_stride,
    out_channel_stride,
    BLOCK_HW: tl.constexpr,
):
    c = tl.program_id(0)
    b = tl.program_id(1)

    w_base = weight_ptr + b * w_batch_stride + c * w_channel_stride
    w0 = tl.load(w_base)
    w1 = tl.load(w_base + w_group_stride)

    for start in tl.range(0, HW, BLOCK_HW):
        offs = start + tl.arange(0, BLOCK_HW)
        mask = offs < HW
        x_base = x_ptr + b * x_batch_stride + c * x_channel_stride + offs
        x0 = tl.load(x_base, mask=mask, other=0.0)
        x1 = tl.load(x_base + x_group_stride, mask=mask, other=0.0)
        out = x0 * w0 + x1 * w1
        out_base = out_ptr + b * out_batch_stride + c * out_channel_stride + offs
        tl.store(out_base, out, mask=mask)


@torch.fx.wrap
def fused_broadcast_mul_sum_contiguous(weight, x):
    b = x.shape[0]
    c = x.shape[2]
    h = x.shape[3]
    w = x.shape[4]
    hw = h * w

    out = torch.empty((b, c, h, w), device=x.device, dtype=x.dtype)
    grid = lambda META: (c, b)
    _weighted_sum2_kernel[grid](
        weight,
        x,
        out,
        hw,
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        x.stride(0),
        x.stride(1),
        x.stride(2),
        out.stride(0),
        out.stride(1),
    )
    return out


def replacement_func():
    return fused_broadcast_mul_sum_contiguous