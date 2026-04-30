import torch
import triton
import triton.language as tl
from pass_dir.fused_depthwise_conv_add_permute_common import fused_dispatch


def pattern(in_0, in_1, in_2):
    conv2d = torch.conv2d(in_2, in_0, None, (1, 1), (32, 0), (1, 1), 4)
    in_1 += conv2d
    tmp_3 = in_1.permute(0, 2, 1, 3)
    tmp_4 = tmp_3.contiguous()
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, "conv_add_permute")


def replacement_func():
    return fused_dispatch