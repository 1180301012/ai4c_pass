import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_C": 4, "BLOCK_HW": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_C": 8, "BLOCK_HW": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_C": 4, "BLOCK_HW": 512}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_C": 8, "BLOCK_HW": 512}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_C": 4, "BLOCK_HW": 1024}, num_warps=8, num_stages=2),
    ],
    key=["C", "HW"],
)
@triton.jit
def _fused_softmax_weighted_sum_kernel(
    x_ptr,
    logits_ptr,
    out_ptr,
    C,
    HW,
    x_batch_stride,
    x_group_stride,
    x_channel_stride,
    logits_batch_stride,
    logits_group_stride,
    logits_channel_stride,
    out_batch_stride,
    out_channel_stride,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_hw = tl.program_id(0)
    pid_c = tl.program_id(1)
    b = tl.program_id(2)

    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask_c = offs_c < C
    mask = mask_c[:, None] & (offs_hw[None, :] < HW)

    logits_base = logits_ptr + b * logits_batch_stride + offs_c * logits_channel_stride
    logit0 = tl.load(logits_base, mask=mask_c, other=0.0).to(tl.float32)
    logit1 = tl.load(logits_base + logits_group_stride, mask=mask_c, other=0.0).to(tl.float32)
    delta = logit1 - logit0
    w1 = 1.0 / (1.0 + tl.exp(-delta))

    x_base = x_ptr + b * x_batch_stride + offs_c[:, None] * x_channel_stride + offs_hw[None, :]
    x0 = tl.load(x_base, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x_base + x_group_stride, mask=mask, other=0.0).to(tl.float32)
    out = x0 + w1[:, None] * (x1 - x0)

    out_base = out_ptr + b * out_batch_stride + offs_c[:, None] * out_channel_stride + offs_hw[None, :]
    tl.store(out_base, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_C": 4, "BLOCK_HW": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_C": 8, "BLOCK_HW": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_C": 4, "BLOCK_HW": 512}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_C": 8, "BLOCK_HW": 512}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_C": 4, "BLOCK_HW": 1024}, num_warps=8, num_stages=2),
    ],
    key=["C", "HW"],
)
@triton.jit
def _tail_weighted_sum_kernel(
    weight_ptr,
    x_ptr,
    out_ptr,
    C,
    HW,
    w_batch_stride,
    w_group_stride,
    w_channel_stride,
    x_batch_stride,
    x_group_stride,
    x_channel_stride,
    out_batch_stride,
    out_channel_stride,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_hw = tl.program_id(0)
    pid_c = tl.program_id(1)
    b = tl.program_id(2)

    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask_c = offs_c < C
    mask = mask_c[:, None] & (offs_hw[None, :] < HW)

    w_base = weight_ptr + b * w_batch_stride + offs_c * w_channel_stride
    w0 = tl.load(w_base, mask=mask_c, other=0.0).to(tl.float32)
    w1 = tl.load(w_base + w_group_stride, mask=mask_c, other=0.0).to(tl.float32)

    x_base = x_ptr + b * x_batch_stride + offs_c[:, None] * x_channel_stride + offs_hw[None, :]
    x0 = tl.load(x_base, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x_base + x_group_stride, mask=mask, other=0.0).to(tl.float32)
    out = x0 * w0[:, None] + x1 * w1[:, None]

    out_base = out_ptr + b * out_batch_stride + offs_c[:, None] * out_channel_stride + offs_hw[None, :]
    tl.store(out_base, out, mask=mask)


@torch.fx.wrap
def fused_dispatch(arg0, arg1, route):
    if route == 0:
        in_0 = arg0
        in_1 = arg1
        b = in_0.shape[0]
        c = in_0.shape[2]
        h = in_0.shape[3]
        w = in_0.shape[4]
        hw = h * w
        out = torch.empty((b, c, h, w), device=in_0.device, dtype=in_0.dtype)
        grid = lambda META: (triton.cdiv(hw, META["BLOCK_HW"]), triton.cdiv(c, META["BLOCK_C"]), b)
        _fused_softmax_weighted_sum_kernel[grid](
            in_0,
            in_1,
            out,
            c,
            hw,
            in_0.stride(0),
            in_0.stride(1),
            in_0.stride(2),
            in_1.stride(0),
            in_1.stride(1),
            in_1.stride(3),
            out.stride(0),
            out.stride(1),
        )
        return out

    weight = arg0
    x = arg1
    b = x.shape[0]
    c = x.shape[2]
    h = x.shape[3]
    w = x.shape[4]
    hw = h * w
    out = torch.empty((b, c, h, w), device=x.device, dtype=x.dtype)
    grid = lambda META: (triton.cdiv(hw, META["BLOCK_HW"]), triton.cdiv(c, META["BLOCK_C"]), b)
    _tail_weighted_sum_kernel[grid](
        weight,
        x,
        out,
        c,
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
    return fused_dispatch