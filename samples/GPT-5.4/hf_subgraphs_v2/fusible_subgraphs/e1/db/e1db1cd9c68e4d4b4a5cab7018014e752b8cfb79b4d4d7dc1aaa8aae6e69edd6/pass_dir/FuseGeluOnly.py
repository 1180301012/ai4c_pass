import torch

from pass_dir.shared_gelu_mean_kernels import shared_dispatch


def pattern(in_0):
    tmp_0 = torch.nn.functional.gelu(in_0)
    return tmp_0


def replacement_args(in_0):
    return (in_0, "gelu")


def replacement_func():
    return shared_dispatch