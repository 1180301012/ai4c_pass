import torch
from pass_dir.shared_fused_gelu_transpose_add_layernorm import fused_dispatch


def pattern(x, weight, bias):
    tmp = torch.nn.functional.layer_norm(x, (1024,), weight, bias, 1e-05)
    return tmp


def replacement_args(x, weight, bias):
    return (x, weight, bias, "layernorm_1024")


def replacement_func():
    return fused_dispatch