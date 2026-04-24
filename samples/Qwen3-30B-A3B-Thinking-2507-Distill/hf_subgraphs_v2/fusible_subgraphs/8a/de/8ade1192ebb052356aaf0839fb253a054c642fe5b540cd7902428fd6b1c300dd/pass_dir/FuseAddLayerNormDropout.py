import torch
from pass_dir.shared_ln_kernel import layer_norm_triton


def pattern(x, weight, bias):
    out = torch.nn.functional.layer_norm(x, (1024,), weight, bias, 1e-05)
    return out


def replacement_args(x, weight, bias):
    return (x, weight, bias)


def replacement_func():
    return layer_norm_triton