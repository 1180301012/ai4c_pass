import torch
from pass_dir.shared_kernel import shared_dispatch


def pattern(in_5, in_1, in_0):
    linear = torch.nn.functional.linear(in_5, in_1, in_0)
    return linear


def replacement_args(in_5, in_1, in_0):
    return (in_5, in_1, in_0, "route_linear1")


def replacement_func():
    return shared_dispatch