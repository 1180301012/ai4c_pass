import torch
from pass_dir.fused_add_mean_common import dispatch_fused_add_mean


def pattern(in_0, in_1, in_2):
    tmp_0 = in_1 + in_2
    tmp_0 += in_0
    tmp_1 = tmp_0
    tmp_2 = tmp_1.mean((2, 3), keepdim=True)
    return (tmp_1, tmp_2)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, "ternary")


def replacement_func():
    return dispatch_fused_add_mean