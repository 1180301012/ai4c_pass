import torch
import triton
import triton.language as tl
from pass_dir.shared_dispatch import shared_dispatch


def pattern(in_1, in_2, in_3):
    tmp_1 = in_1 - in_2
    tmp_2 = tmp_1.pow(2)
    tmp_3 = tmp_2.sum(dim=3)
    return in_3 * tmp_3


def replacement_args(in_1, in_2, in_3):
    return (in_1, in_2, "sub_pow_sum_mul", in_3)


def replacement_func():
    return shared_dispatch