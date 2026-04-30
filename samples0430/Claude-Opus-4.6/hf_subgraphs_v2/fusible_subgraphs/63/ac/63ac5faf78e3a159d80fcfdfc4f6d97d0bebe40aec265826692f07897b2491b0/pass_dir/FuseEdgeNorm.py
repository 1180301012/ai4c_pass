import torch
from pass_dir.shared_kernels import dispatch


def pattern(tmp_2, in_2, in_4, in_5):
    tmp_5 = tmp_2[in_5]
    tmp_6 = tmp_5 * in_4
    tmp_7 = tmp_2[in_2]
    tmp_8 = tmp_6 * tmp_7
    return tmp_8


def replacement_args(tmp_2, in_2, in_4, in_5):
    return (tmp_2, in_5, in_2, in_4, "edge_norm")


def replacement_func():
    return dispatch