import torch
from pass_dir.layer_norm_triton_kernel import triton_layer_norm_dispatch


def pattern(x, weight, bias):
    return torch.nn.functional.layer_norm(x, (192,), weight, bias, 1e-06)


def replacement_args(x, weight, bias):
    return (x, weight, bias, "192")


def replacement_func():
    return triton_layer_norm_dispatch