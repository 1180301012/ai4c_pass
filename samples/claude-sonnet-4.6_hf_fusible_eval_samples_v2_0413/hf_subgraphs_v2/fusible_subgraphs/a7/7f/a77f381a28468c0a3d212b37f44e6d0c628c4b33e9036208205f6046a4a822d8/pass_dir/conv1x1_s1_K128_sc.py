import torch
from pass_dir.conv1x1_shared import conv1x1_dispatch


def pattern(in_0, in_1):
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_2 = conv2d[(slice(None, None, None), slice(None, 128, None), slice(None, None, None), slice(None, None, None))]
    return (tmp_2, conv2d)


def replacement_args(in_0, in_1):
    return (in_0, in_1, "s1_K128_sc")


def replacement_func():
    return conv1x1_dispatch