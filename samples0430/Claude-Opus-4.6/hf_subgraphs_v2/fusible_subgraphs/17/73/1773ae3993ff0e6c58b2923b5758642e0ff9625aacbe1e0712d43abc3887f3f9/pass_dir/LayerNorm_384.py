import torch
from pass_dir._shared import triton_ln_dispatch


def pattern(x, weight, bias):
    return torch.nn.functional.layer_norm(x, (384,), weight, bias, 1e-12)


def replacement_args(x, weight, bias):
    return (x, weight, bias, "ln_384")


def replacement_func():
    return triton_ln_dispatch