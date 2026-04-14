import torch
from pass_dir.cca_kernels import cca_dispatch


def pattern(in_0, in_2, in_5):
    tmp_3 = in_5 * in_0
    tmp_4 = tmp_3 + in_2
    tmp_5 = tmp_4.contiguous()
    return tmp_5


def replacement_args(in_0, in_2, in_5):
    return (in_0, in_2, in_5, "scale_add")


def replacement_func():
    return cca_dispatch