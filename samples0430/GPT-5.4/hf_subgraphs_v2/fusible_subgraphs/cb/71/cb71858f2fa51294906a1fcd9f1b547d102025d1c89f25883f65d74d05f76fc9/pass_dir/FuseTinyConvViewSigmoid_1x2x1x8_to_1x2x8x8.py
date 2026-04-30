import torch
import triton
import triton.language as tl

from pass_dir.shared_kernels import shared_replacement_dispatch


def pattern(in_0, in_1, in_2):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d.view(1, 2, 8, 8)
    tmp_4 = tmp_3.sigmoid()
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, "tiny_conv_view_sigmoid")


def replacement_func():
    return shared_replacement_dispatch