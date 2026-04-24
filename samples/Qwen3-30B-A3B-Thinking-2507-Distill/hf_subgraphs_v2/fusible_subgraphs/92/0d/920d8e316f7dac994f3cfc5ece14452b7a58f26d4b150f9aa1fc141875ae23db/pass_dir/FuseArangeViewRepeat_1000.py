import torch
from pass_dir.shared_kernels import _arange_dispatch


def pattern(x):
    tmp_1 = x.view(1, -1)
    tmp_2 = tmp_1.repeat(2, 1)
    return tmp_2


def replacement_args(x):
    return (x, "N/A")


def replacement_func():
    return _arange_dispatch