import torch
from pass_dir._shared_ln_kernel import layernorm_triton_dispatch


def pattern(x, weight, bias):
    tmp_8 = torch.nn.functional.layer_norm(x, (16,), weight, bias, 1e-05)
    tmp_9 = torch.nn.functional.dropout(tmp_8, 0.0, False, False)
    return tmp_9


def replacement_args(x, weight, bias):
    return (x, weight, bias, "c16")


def replacement_func():
    return layernorm_triton_dispatch