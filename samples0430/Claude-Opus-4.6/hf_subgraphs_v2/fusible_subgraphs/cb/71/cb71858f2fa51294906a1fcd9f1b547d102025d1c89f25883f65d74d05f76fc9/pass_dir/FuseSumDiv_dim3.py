import torch
from pass_dir.shared_dispatch import dispatch


def pattern(x):
    s = x.sum(dim=3, keepdim=True)
    out = x / s
    return out


def replacement_args(x):
    return (x, x, x, "norm")


def replacement_func():
    return dispatch