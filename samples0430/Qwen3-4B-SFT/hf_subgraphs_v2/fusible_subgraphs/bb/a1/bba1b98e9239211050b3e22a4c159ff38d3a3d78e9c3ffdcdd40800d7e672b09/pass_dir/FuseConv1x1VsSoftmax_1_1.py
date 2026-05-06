import torch
from pass_dir.fused_conv1x1_view_softmax_kernel import fused_conv1x1_view_softmax


def pattern(in_0, in_1, in_2):
    """Match: conv2d -> view(1, 1, -1) -> softmax(dim=-1)"""
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d.view(1, 1, -1)
    tmp_4 = tmp_3.softmax(dim=-1)
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return fused_conv1x1_view_softmax