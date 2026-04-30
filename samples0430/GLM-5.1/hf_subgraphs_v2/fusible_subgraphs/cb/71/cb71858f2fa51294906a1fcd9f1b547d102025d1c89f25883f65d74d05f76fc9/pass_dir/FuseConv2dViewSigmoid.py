import torch
from pass_dir._shared import shared_dispatch_wrapper


def pattern(in_0, in_1, in_2):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d.view(1, 2, 8, 8)
    tmp_4 = tmp_3.sigmoid()
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    return (in_2, in_1, in_0, "conv2d_view_sigmoid")


def replacement_func():
    return shared_dispatch_wrapper