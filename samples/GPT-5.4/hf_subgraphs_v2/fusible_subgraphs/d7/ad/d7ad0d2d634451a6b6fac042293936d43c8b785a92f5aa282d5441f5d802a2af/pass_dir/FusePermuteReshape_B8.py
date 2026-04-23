import torch
from pass_dir.shared_permute_reshape import shared_permute_reshape_dispatch


def pattern(x):
    tmp_3 = x.permute(0, 2, 1)
    tmp_4 = tmp_3.reshape(8, -1, 16, 16)
    return tmp_4


def replacement_args(x):
    return (x, "b8")


def replacement_func():
    return shared_permute_reshape_dispatch