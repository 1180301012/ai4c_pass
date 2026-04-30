import torch
from pass_dir.shared_kernels import _unified_dispatch


def pattern(x, weight, bias):
    return torch.nn.functional.linear(x, weight, bias)


def replacement_args(x, weight, bias):
    return (x, weight, bias, "linear")


def replacement_func():
    return _unified_dispatch