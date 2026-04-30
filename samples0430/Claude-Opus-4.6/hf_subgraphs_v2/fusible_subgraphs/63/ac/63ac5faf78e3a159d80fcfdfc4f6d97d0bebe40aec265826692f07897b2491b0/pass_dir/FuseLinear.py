import torch
from pass_dir.shared_kernels import dispatch


def pattern(x, w):
    return torch.nn.functional.linear(x, w, None)


def replacement_args(x, w):
    return (x, w, x, x, "linear")


def replacement_func():
    return dispatch