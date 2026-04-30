import torch
import triton
import triton.language as tl
from pass_dir._shared_conv_hsigmoid_mul import shared_dispatch


def pattern(in_0: torch.Tensor, in_1, in_2, in_3):
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d + 1.0
    tmp_4 = tmp_3 / 2.0
    tmp_5 = tmp_4.clamp_(0.0, 1.0)
    tmp_6 = in_2 * tmp_5
    return tmp_6


def replacement_args(in_0: torch.Tensor, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3, "plus1_div2")


def replacement_func():
    return shared_dispatch