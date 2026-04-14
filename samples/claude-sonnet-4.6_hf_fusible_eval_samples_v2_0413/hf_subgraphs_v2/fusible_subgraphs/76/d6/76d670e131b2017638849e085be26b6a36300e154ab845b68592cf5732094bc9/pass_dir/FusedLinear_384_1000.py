import torch
from pass_dir.shared_dispatch import dispatch


def pattern(input, weight, bias):
    return torch.nn.functional.linear(input, weight, bias)


def replacement_args(input, weight, bias):
    return (input, weight, bias, None, None, "linear")


def replacement_func():
    return dispatch