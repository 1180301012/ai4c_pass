import torch
from pass_dir.layer_norm_kernels import layer_norm_dispatch


def pattern(bias, weight, input):
    result = torch.nn.functional.layer_norm(input, (32,), weight, bias, 1e-12)
    return result


def replacement_args(bias, weight, input):
    return (input, weight, bias, "ln32")


def replacement_func():
    return layer_norm_dispatch