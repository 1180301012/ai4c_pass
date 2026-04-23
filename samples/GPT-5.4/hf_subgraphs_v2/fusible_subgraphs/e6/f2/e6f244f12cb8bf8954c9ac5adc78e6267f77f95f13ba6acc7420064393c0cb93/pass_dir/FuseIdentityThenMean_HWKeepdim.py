import operator
import torch
from pass_dir.shared_fused_add_mean import shared_replacement


def pattern(in_0):
    tmp_0 = 0 + in_0
    tmp_0 = operator.iadd(tmp_0, 0)
    tmp_1 = tmp_0
    tmp_2 = tmp_1.mean((2, 3), keepdim=True)
    return (tmp_1, tmp_2)


def replacement_args(in_0):
    return (in_0, in_0, in_0, "identity")


def replacement_func():
    return shared_replacement