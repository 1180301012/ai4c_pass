import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 4)
    tmp_3 = torch.sigmoid(conv2d)
    tmp_4 = tmp_3.view(1, -1, 1, 1)
    tmp_5 = in_2 * tmp_4
    tmp_6 = tmp_5.contiguous()
    return tmp_6


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_HW": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_HW": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_HW": 512}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_HW": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_HW": 2048}, num_warps=8, num_stages=2),
    ],
    key=["HW"],
)
@triton.jit
def fused_se_mul_contiguous_channel_kernel(
    bias_ptr,
    weight_ptr,
    x_ptr,
    gap_ptr,
    out_ptr,
    C,
    HW,
    BLOCK_HW: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_n = tl.program_id(1)

    acc = tl.load(bias_ptr + pid_c).to(tl.float32)
    group = pid_c // 24
    weight_base = weight_ptr + pid_c * 8
    gap_base = gap_ptr + pid_n * 32 + group * 8

    for k in range(8):
        w = tl.load(weight_base + k).to(tl.float32)
        gv = tl.load(gap_base + k).to(tl.float32)
        acc += w * gv

    gate = 1.0 / (1.0 + tl.exp(-acc))
    base = pid_n * (C * HW) + pid_c * HW

    for start in tl.range(0, HW, BLOCK_HW):
        offs = start + tl.arange(0, BLOCK_HW)
        mask = offs < HW
        x = tl.load(x_ptr + base + offs, mask=mask, other=0).to(tl.float32)
        out = x * gate
        tl.store(out_ptr + base + offs, out, mask=mask)


@triton.jit
def fused_se_mul_strided_channel_kernel(
    bias_ptr,
    bias_stride_0,
    weight_ptr,
    weight_stride_0,
    weight_stride_1,
    x_ptr,
    x_stride_0,
    x_stride_1,
    x_stride_2,
    x_stride_3,
    gap_ptr,
    gap_stride_0,
    gap_stride_1,
    out_ptr,
    out_stride_0,
    out_stride_1,
    out_stride_2,
    out_stride_3,
    W,
    HW,
    BLOCK_HW: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_n = tl.program_id(1)

    acc = tl.load(bias_ptr + pid_c * bias_stride_0).to(tl.float32)
    group = pid_c // 24
    weight_base = weight_ptr + pid_c * weight_stride_0
    gap_base = gap_ptr + pid_n * gap_stride_0 + group * 8 * gap_stride_1

    for k in range(8):
        wv = tl.load(weight_base + k * weight_stride_1).to(tl.float32)
        gv = tl.load(gap_base + k * gap_stride_1).to(tl.float32)
        acc += wv * gv

    gate = 1.0 / (1.0 + tl.exp(-acc))

    for start in tl.range(0, HW, BLOCK_HW):
        offs = start + tl.arange(0, BLOCK_HW)
        mask = offs < HW
        h = offs // W
        w = offs - h * W
        x_ptrs = x_ptr + pid_n * x_stride_0 + pid_c * x_stride_1 + h * x_stride_2 + w * x_stride_3
        out_ptrs = out_ptr + pid_n * out_stride_0 + pid_c * out_stride_1 + h * out_stride_2 + w * out_stride_3
        x = tl.load(x_ptrs, mask=mask, other=0).to(tl.float32)
        out = x * gate
        tl.store(out_ptrs, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_HW": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_HW": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_HW": 512}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_HW": 1024}, num_warps=8, num_stages=2),
    ],
    key=["HW"],
)
@triton.jit
def fused_se_mul_group24_tiled_f32_kernel(
    bias_ptr,
    weight_ptr,
    x_ptr,
    gap_ptr,
    out_ptr,
    C,
    HW,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_hw = tl.program_id(0)
    pid_g = tl.program_id(1)
    pid_n = tl.program_id(2)

    lane_c = tl.arange(0, BLOCK_C)
    offs_c = pid_g * 24 + lane_c
    valid_c = lane_c < 24
    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask_hw = offs_hw < HW
    mask = valid_c[:, None] & mask_hw[None, :]

    acc = tl.load(bias_ptr + offs_c, mask=valid_c, other=0.0).to(tl.float32)
    gap_base = gap_ptr + pid_n * 32 + pid_g * 8
    for k in range(8):
        w = tl.load(weight_ptr + offs_c * 8 + k, mask=valid_c, other=0.0).to(tl.float32)
        gv = tl.load(gap_base + k).to(tl.float32)
        acc += w * gv
    gate = 1.0 / (1.0 + tl.exp(-acc))

    base = pid_n * (C * HW) + offs_c[:, None] * HW + offs_hw[None, :]
    x = tl.load(x_ptr + base, mask=mask, other=0.0)
    out = x * gate[:, None]
    tl.store(out_ptr + base, out, mask=mask)


@torch.fx.wrap
def fused_grouped_se_conv_sigmoid_mul(in_0, in_1, in_2, in_3):
    n, c, h, w = in_2.shape
    hw = h * w
    out = torch.empty_like(in_2)

    if in_0.is_contiguous() and in_1.is_contiguous() and in_2.is_contiguous() and in_3.is_contiguous():
        if in_2.dtype == torch.float32 and hw <= 96 * 96:
            grid = lambda META: (triton.cdiv(hw, META["BLOCK_HW"]), 4, n)
            fused_se_mul_group24_tiled_f32_kernel[grid](
                in_0,
                in_1,
                in_2,
                in_3,
                out,
                c,
                hw,
                BLOCK_C=32,
            )
        else:
            fused_se_mul_contiguous_channel_kernel[(c, n)](
                in_0,
                in_1,
                in_2,
                in_3,
                out,
                c,
                hw,
            )
    else:
        fused_se_mul_strided_channel_kernel[(c, n)](
            in_0,
            in_0.stride(0),
            in_1,
            in_1.stride(0),
            in_1.stride(1),
            in_2,
            in_2.stride(0),
            in_2.stride(1),
            in_2.stride(2),
            in_2.stride(3),
            in_3,
            in_3.stride(0),
            in_3.stride(1),
            out,
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            w,
            hw,
            BLOCK_HW=256,
            num_warps=4,
            num_stages=2,
        )
    return out


def replacement_func():
    return fused_grouped_se_conv_sigmoid_mul