import torch
from pass_dir.kernel_defs import _shared_dispatch


def pattern(in_2):
    tmp_4 = in_2.mean(dim=-2, keepdim=True)
    return tmp_4


def replacement_args(in_2):
    return (in_2, "mean_dim1")


def replacement_func():
    return _shared_dispatch