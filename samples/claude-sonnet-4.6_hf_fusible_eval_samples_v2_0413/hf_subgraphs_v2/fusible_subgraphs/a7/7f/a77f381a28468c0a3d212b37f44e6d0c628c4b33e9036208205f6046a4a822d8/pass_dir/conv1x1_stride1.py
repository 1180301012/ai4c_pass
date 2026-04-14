import torch
from pass_dir.conv1x1_shared import conv1x1_route


def pattern(in_0, in_1):
    return torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)


def replacement_args(in_0, in_1):
    return (in_0, in_1, "s1")


def replacement_func():
    return conv1x1_route