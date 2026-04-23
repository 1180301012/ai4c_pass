import torch
import triton
import triton.language as tl

from pass_dir.shared_kernels import replacement_dispatch


def pattern(in_4, tmp_3, tmp_2):
    tmp_4 = torch.conv2d(in_4, tmp_3, tmp_2, (1, 1), (1, 1), (1, 1), 1024)
    tmp_5 = tmp_4 + in_4
    return tmp_5


def replacement_args(in_4, tmp_3, tmp_2):
    return (in_4, tmp_3, tmp_2, "dwconv_residual")


def replacement_func():
    return replacement_dispatch