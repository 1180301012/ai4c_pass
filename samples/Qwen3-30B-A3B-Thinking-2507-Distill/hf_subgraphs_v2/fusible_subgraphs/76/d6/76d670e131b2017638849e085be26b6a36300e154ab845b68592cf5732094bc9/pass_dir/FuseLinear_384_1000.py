import torch
from pass_dir.shared_kernels import shared_dispatch


def pattern(input, weight, bias):
    return torch.nn.functional.linear(input, weight, bias)


def replacement_args(input, weight, bias):
    return (input, weight, bias, "linear")


def replacement_func():
    return shared_dispatch