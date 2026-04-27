import torch
from pass_dir.shared_dispatch import dispatch


def pattern(x):
    return x.transpose(-2, -1)


def replacement_args(x):
    return (x, "transpose")


def replacement_func():
    return dispatch