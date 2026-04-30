import torch
from pass_dir.shared_ops import fused_dispatch


def pattern(in_2, in_1, in_0):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d.view(1, 2, 8, 8)
    tmp_4 = tmp_3.sigmoid()
    return tmp_4


def replacement_args(in_2, in_1, in_0):
    # Route string appended as last arg for shared dispatch wrapper
    return (in_2, in_1, in_0, "conv_sigm")


def replacement_func():
    return fused_dispatch