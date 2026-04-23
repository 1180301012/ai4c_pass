import torch

from pass_dir.shared_gelu_mean_kernels import shared_dispatch


def pattern(in_0):
    tmp_0 = in_0.mean((2, 3), keepdim=True)
    return tmp_0


def replacement_args(in_0):
    return (in_0, "mean_hw_keepdim")


def replacement_func():
    return shared_dispatch