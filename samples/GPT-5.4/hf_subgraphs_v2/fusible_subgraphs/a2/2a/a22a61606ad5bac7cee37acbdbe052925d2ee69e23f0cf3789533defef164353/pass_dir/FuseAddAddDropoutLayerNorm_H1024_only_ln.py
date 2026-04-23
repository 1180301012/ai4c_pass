import torch
from pass_dir.shared_embedding_ln import fused_add_dropout_layernorm


def pattern(a, b, c, gamma, beta):
    t = a + b
    t += c
    d = torch.nn.functional.dropout(t, 0.1, False, False)
    y = torch.nn.functional.layer_norm(d, (1024,), gamma, beta, 1e-12)
    return (y,)


def replacement_args(a, b, c, gamma, beta):
    return (a, b, c, gamma, beta, False)


def replacement_func():
    return fused_add_dropout_layernorm