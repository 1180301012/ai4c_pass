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
        triton.Config({"BLOCK_HW": 64, "BLOCK_C": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_HW": 128, "BLOCK_C": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_HW": 256, "BLOCK_C": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_HW": 128, "BLOCK_C": 32}, num_warps=8, num_stages=3),
    ],
    key=["HW", "C"],
)
@triton.jit
def fused_grouped_conv_sigmoid_mul_contiguous_kernel(
    x_ptr,
    bias_ptr,
    weight_ptr,
    gap_ptr,
    out_ptr,
    HW,
    C,
    x_stride_c,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_hw = tl.program_id(0)
    pid_c = tl.program_id(1)

    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    mask_hw = offs_hw < HW
    mask_c = offs_c < C

    acc = tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)
    group_idx = offs_c // 24
    base_in = group_idx * 8

    for k in range(8):
        w = tl.load(weight_ptr + offs_c * 8 + k, mask=mask_c, other=0.0).to(tl.float32)
        v = tl.load(gap_ptr + base_in + k, mask=mask_c, other=0.0).to(tl.float32)
        acc += w * v

    gate = tl.sigmoid(acc)

    x_offsets = offs_c[:, None] * x_stride_c + offs_hw[None, :]
    out_offsets = offs_c[:, None] * HW + offs_hw[None, :]
    mask = mask_c[:, None] & mask_hw[None, :]

    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
    y = x.to(tl.float32) * gate[:, None]
    tl.store(out_ptr + out_offsets, y, mask=mask)


@torch.fx.wrap
def fused_grouped_conv_sigmoid_mul_contiguous(in_0, in_1, in_2, in_3):
    hw = in_2.shape[2] * in_2.shape[3]
    c = in_2.shape[1]

    out = torch.empty(in_2.shape, device=in_2.device, dtype=in_2.dtype)

    grid = lambda meta: (triton.cdiv(hw, meta["BLOCK_HW"]), triton.cdiv(c, meta["BLOCK_C"]))
    fused_grouped_conv_sigmoid_mul_contiguous_kernel[grid](
        x_ptr=in_2,
        bias_ptr=in_0,
        weight_ptr=in_1,
        gap_ptr=in_3,
        out_ptr=out,
        HW=hw,
        C=c,
        x_stride_c=in_2.stride(1),
    )
    return out


def replacement_func():
    return fused_grouped_conv_sigmoid_mul_contiguous