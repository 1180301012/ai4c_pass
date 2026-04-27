import torch
from pass_dir.ln_kernels import triton_ln_dispatch


def pattern(x, weight, bias):
    return torch.nn.functional.layer_norm(x, (192,), weight, bias, 1e-06)


def replacement_args(x, weight, bias):
    return (x, weight, bias, "n192")


def replacement_func():
    return triton_ln_dispatch