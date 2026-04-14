import torch
from pass_dir.dw_conv_mean_kernel import fused_dw_conv_mean


def pattern(in_0, in_1):
    conv2d = torch.conv2d(in_1, in_0, None, (2, 2), (1, 1), (1, 1), 256)
    tmp_2 = conv2d.mean((2, 3), keepdim=True)
    return (conv2d, tmp_2)


def replacement_args(in_0, in_1):
    return (in_0, in_1, "s2")


def replacement_func():
    return fused_dw_conv_mean