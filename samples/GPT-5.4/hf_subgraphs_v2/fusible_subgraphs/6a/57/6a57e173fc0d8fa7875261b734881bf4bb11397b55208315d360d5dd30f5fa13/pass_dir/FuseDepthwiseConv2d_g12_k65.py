import torch
from pass_dir.shared_depthwise_conv_only import shared_depthwise_conv_replacement


def pattern(in_0, in_2):
    conv2d = torch.conv2d(in_2, in_0, None, (1, 1), (32, 0), (1, 1), 12)
    return conv2d


def replacement_args(in_0, in_2):
    return (in_2, in_0)


def replacement_func():
    return shared_depthwise_conv_replacement