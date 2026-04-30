import torch
from pass_dir.shared_kernels import shared_dispatch


def pattern(x, scale):
    out = x * scale
    return out


def replacement_args(x, scale):
    return (x, scale, "row_scale")


def replacement_func():
    return shared_dispatch