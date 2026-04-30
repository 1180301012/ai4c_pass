import torch
from pass_dir.ln_dispatch_kernel import _dispatch


def pattern(x, weight, bias):
    return torch.nn.functional.layer_norm(x, (768,), weight, bias, 1e-05)


def replacement_args(x, weight, bias):
    return (x, weight, bias, "C768")


def replacement_func():
    return _dispatch