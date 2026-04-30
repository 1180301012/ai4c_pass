import torch
from pass_dir.shared_dispatch import shared_replacement_dispatch


def pattern(x):
    tmp_11 = x.mean((2, 3), keepdim=True)
    return tmp_11


def replacement_args(x):
    return (x, "mean_only")


def replacement_func():
    return shared_replacement_dispatch