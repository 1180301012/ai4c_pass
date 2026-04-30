import torch
from pass_dir.shared_kernel import fused_dispatch


def pattern(x):
    gelu_out = torch.nn.functional.gelu(x, approximate='none')
    drop_out = torch.nn.functional.dropout(gelu_out, 0.0, False, False)
    return drop_out


def replacement_args(x):
    return (x, x, x, "simple_gelu")


def replacement_func():
    return fused_dispatch