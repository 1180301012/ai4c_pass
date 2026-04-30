import torch
from pass_dir.shared_kernels import shared_dispatch


def pattern(feat, weight, bias):
    conv_out = torch.conv2d(feat, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    return conv_out


def replacement_args(feat, weight, bias):
    return (feat, weight, bias, "conv")


def replacement_func():
    return shared_dispatch