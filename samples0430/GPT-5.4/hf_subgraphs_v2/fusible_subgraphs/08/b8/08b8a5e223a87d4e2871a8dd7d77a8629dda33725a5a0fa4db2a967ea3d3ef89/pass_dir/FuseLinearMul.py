import torch
from pass_dir.shared_kernels import shared_dispatch


def pattern(w, x, y):
    linear = torch.nn.functional.linear(x, w, None)
    out = y * linear
    return out


def replacement_args(w, x, y):
    return (w, x, y, "linear_mul")


def replacement_func():
    return shared_dispatch