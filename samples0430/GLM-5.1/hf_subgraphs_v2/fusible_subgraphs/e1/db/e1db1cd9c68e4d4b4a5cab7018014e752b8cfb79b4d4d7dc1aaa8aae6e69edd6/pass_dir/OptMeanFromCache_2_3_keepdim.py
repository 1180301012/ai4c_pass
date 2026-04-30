import torch
from pass_dir.shared_fused_kernel import fused_gelu_mean_dispatch


def pattern(in_0):
    tmp_1 = in_0.mean((2, 3), keepdim=True)
    return tmp_1


def replacement_args(in_0):
    return (in_0, "mean_from_cache")


def replacement_func():
    return fused_gelu_mean_dispatch