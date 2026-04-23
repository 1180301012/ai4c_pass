import torch
from pass_dir.shared_zero_lazy import shared_replacement_func


def pattern(x):
    out = x.new_zeros((128, 128))
    return out


def replacement_args(x):
    return (x, 1)


def replacement_func():
    return shared_replacement_func()