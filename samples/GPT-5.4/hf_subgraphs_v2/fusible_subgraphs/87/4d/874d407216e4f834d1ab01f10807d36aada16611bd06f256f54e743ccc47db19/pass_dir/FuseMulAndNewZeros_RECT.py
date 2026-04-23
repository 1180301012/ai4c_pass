import torch
from pass_dir.shared_fused_mul_zero import shared_replacement_func


def pattern(in_1, in_2):
    tmp_0 = in_1.view(-1, 1)
    tmp_1 = tmp_0 * in_2
    tmp_2 = tmp_1.new_zeros((128, 128))
    return tmp_1, tmp_2


def replacement_args(in_1, in_2):
    return (in_1, in_2, "rect")


def replacement_func():
    return shared_replacement_func()