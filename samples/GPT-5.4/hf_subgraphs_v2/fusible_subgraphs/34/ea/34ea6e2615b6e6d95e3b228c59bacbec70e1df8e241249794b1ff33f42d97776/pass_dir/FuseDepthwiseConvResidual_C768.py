import torch
import triton
import triton.language as tl

from pass_dir.shared_kernels import replacement_dispatch


def pattern(in_4, in_3, in_2):
    conv2d = torch.conv2d(in_4, in_3, in_2, (1, 1), (1, 1), (1, 1), 768)
    tmp_5 = conv2d + in_4
    return tmp_5


def replacement_args(in_4, in_3, in_2):
    return (in_4, in_3, in_2, "dwconv_residual")


def replacement_func():
    return replacement_dispatch