import torch
from pass_dir.shared_layernorm import fused_dropout_layernorm


def pattern(x, gamma, beta):
    d = torch.nn.functional.dropout(x, 0.1, False, False)
    y = torch.nn.functional.layer_norm(d, (64,), gamma, beta, 1e-12)
    return (y,)


def replacement_args(x, gamma, beta):
    return (x, gamma, beta, False)


def replacement_func():
    return fused_dropout_layernorm