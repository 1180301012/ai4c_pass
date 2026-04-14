import torch
import triton
import triton.language as tl
from pass_dir.interp_kernel import fused_linear_view


def pattern(in_0, in_1, in_2):
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.nn.functional.linear(in_2, tmp_1, tmp_0)
    tmp_3 = tmp_2.permute(0, 2, 1)
    tmp_4 = tmp_3.reshape(2, -1, 16, 16)
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return fused_linear_view