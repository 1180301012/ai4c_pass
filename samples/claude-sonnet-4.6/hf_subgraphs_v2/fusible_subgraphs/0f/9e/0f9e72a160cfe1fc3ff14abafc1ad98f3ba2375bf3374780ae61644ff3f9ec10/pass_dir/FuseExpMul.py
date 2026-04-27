import torch
from pass_dir.shared_kernels import shared_dispatch


def pattern(scale, x):
    tmp_5 = scale.exp()
    tmp_6 = tmp_5 * x
    return tmp_6


def replacement_args(scale, x):
    return (scale, x, "exp_mul")


def replacement_func():
    return shared_dispatch