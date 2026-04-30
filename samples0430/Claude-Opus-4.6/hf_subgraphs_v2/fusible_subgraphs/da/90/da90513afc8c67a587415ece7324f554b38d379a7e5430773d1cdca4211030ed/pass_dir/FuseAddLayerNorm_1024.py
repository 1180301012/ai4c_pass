import torch
from pass_dir.kernels import fused_add_layernorm


def pattern(bias, weight, x, y):
    tmp = x + y
    out = torch.nn.functional.layer_norm(tmp, (1024,), weight, bias, 1e-05)
    r = torch.rand([])
    return out


def replacement_args(bias, weight, x, y):
    return (x, y, weight, bias)


def replacement_func():
    return fused_add_layernorm