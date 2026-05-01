import torch
from pass_dir.shared_fused_ln import fused_gelu_add_ln_dispatch


def pattern(in_0, in_1, x):
    return torch.nn.functional.layer_norm(x, (256,), in_1, in_0, 1e-06)


def replacement_args(in_0, in_1, x):
    return (in_0, in_1, x)


def replacement_func():
    return fused_gelu_add_ln_dispatch