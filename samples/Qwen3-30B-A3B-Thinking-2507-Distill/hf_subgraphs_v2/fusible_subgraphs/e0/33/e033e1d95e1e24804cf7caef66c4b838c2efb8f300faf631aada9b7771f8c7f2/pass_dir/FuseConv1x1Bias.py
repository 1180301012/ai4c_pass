import torch
from pass_dir.kernel_defs import _shared_dispatch


def pattern(in_3, in_1, in_0):
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    return conv2d


def replacement_args(in_3, in_1, in_0):
    return (in_3, in_1, in_0, "conv1x1_bias")


def replacement_func():
    return _shared_dispatch