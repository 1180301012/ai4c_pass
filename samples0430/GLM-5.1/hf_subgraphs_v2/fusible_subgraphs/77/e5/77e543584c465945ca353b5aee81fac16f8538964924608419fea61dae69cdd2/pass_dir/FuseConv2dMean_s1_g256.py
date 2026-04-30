import torch
from pass_dir.fused_conv2d_mean_kernels import fused_conv2d_mean_dispatch


def pattern(weight, input):
    conv2d = torch.conv2d(input, weight, None, (1, 1), (1, 1), (1, 1), 256)
    tmp_2 = conv2d.mean((2, 3), keepdim=True)
    return (conv2d, tmp_2)


def replacement_args(weight, input):
    return (weight, input, "s1_g256")


def replacement_func():
    return fused_conv2d_mean_dispatch