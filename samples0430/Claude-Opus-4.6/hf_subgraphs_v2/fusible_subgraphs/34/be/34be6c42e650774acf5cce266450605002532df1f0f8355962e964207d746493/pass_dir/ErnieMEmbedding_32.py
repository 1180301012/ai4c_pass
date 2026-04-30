import torch
from pass_dir.shared_kernel import fused_add_layernorm_dispatch


def pattern(a, b, weight, bias):
    c = a + b
    d = torch.nn.functional.layer_norm(c, (32,), weight, bias, 1e-05)
    e = torch.nn.functional.dropout(d, 0.1, False, False)
    return e


def replacement_args(a, b, weight, bias):
    return (a, b, weight, bias, "route_32")


def replacement_func():
    return fused_add_layernorm_dispatch