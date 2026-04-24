import torch
import operator
from pass_dir.fused_kernel_impl import fused_conv_add_permute_view


def pattern(weight, in_1, in_2):
    conv = torch.conv2d(in_2, weight, None, (1, 1), (32, 0), (1, 1), 4)
    in_1 += conv
    tmp = in_1.permute(0, 2, 1, 3)
    out = tmp.contiguous()
    return out.view(8, 256, 32)


def replacement_args(weight, in_1, in_2):
    return (weight, in_1, in_2)


def replacement_func():
    return fused_conv_add_permute_view