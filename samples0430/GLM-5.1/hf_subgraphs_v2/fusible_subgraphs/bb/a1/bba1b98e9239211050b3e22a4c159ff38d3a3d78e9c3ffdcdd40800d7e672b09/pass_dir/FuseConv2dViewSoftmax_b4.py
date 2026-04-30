import torch
from pass_dir.fused_kernel import _dispatch


def pattern(in_0, in_1, in_2):
    conv_out = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    viewed = conv_out.view(4, 1, -1)
    result = viewed.softmax(dim=-1)
    return result


def replacement_args(in_0, in_1, in_2):
    return (in_2, in_1, in_0, "b4")


def replacement_func():
    return _dispatch