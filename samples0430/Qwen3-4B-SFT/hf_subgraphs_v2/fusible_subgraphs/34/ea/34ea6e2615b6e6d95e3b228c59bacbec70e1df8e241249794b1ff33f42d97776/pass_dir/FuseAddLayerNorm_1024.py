import torch
from pass_dir.shared_layer_norm import shared_layer_norm_dispatch


def pattern(x, weight, bias):
    out = torch.nn.functional.layer_norm(x, (1024,), weight, bias, 1e-05)
    return out


def replacement_args(x, weight, bias):
    return (x, weight, bias, "C1024")


def replacement_func():
    return shared_layer_norm_dispatch