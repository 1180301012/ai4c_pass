"""
Variant pass: aten-level pattern with tuple args AND no divisor_override arg
(Dynamo may elide the trailing None default completely from the node args).
"""
import torch
import triton
import triton.language as tl

from pass_dir.FuseReluAvgPoolChannelScale import (
    fused_relu_avgpool_scale_kernel,
    copy_channel_unsqueeze_kernel,
)


def pattern(in_0, in_1, in_2):
    tmp_2 = torch.ops.aten.relu.default(in_2)
    tmp_3 = torch.ops.aten.avg_pool2d.default(
        tmp_2, [3, 3], [1, 1], [1, 1], False, False
    )
    tmp_4 = torch.ops.aten.sub.Tensor(tmp_3, tmp_2)
    tmp_5 = torch.ops.aten.unsqueeze.default(in_0, -1)
    tmp_6 = torch.ops.aten.unsqueeze.default(tmp_5, -1)
    tmp_7 = torch.ops.aten.mul.Tensor(tmp_6, tmp_4)
    tmp_8 = torch.ops.aten.add.Tensor(tmp_2, tmp_7)
    tmp_9 = torch.ops.aten.unsqueeze.default(in_1, -1)
    tmp_10 = torch.ops.aten.unsqueeze.default(tmp_9, -1)
    return (tmp_8, tmp_10)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@torch.fx.wrap
def fused_relu_pool_scale_v3(in_0, in_1, in_2):
    B, C, H, W = in_2.shape

    out1 = torch.empty_like(in_2)
    out2 = torch.empty((C, 1, 1), dtype=in_1.dtype, device=in_1.device)

    stride_b = C * H * W
    stride_c = H * W
    stride_h = W
    HW = H * W

    grid_main = lambda meta: (B * C, triton.cdiv(HW, meta['BLOCK_SIZE']))
    fused_relu_avgpool_scale_kernel[grid_main](
        in_2, in_0, out1,
        B, C, H, W,
        stride_b, stride_c, stride_h,
    )

    BLOCK_C = 64
    grid_c = (triton.cdiv(C, BLOCK_C),)
    copy_channel_unsqueeze_kernel[grid_c](in_1, out2, C, BLOCK_C)

    return (out1, out2)


def replacement_func():
    return fused_relu_pool_scale_v3