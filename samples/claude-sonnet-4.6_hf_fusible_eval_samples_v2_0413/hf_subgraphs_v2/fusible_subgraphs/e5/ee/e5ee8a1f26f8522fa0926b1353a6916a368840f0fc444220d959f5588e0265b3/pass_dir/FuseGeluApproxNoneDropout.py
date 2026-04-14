import torch
from pass_dir.shared_gelu_kernel import triton_gelu_dispatch


def pattern(x):
    tmp_3 = torch.nn.functional.gelu(x, approximate='none')
    tmp_4 = torch.nn.functional.dropout(tmp_3, 0.0, False, False)
    return tmp_4


def replacement_args(x):
    return (x, "gelu_approx_none")


def replacement_func():
    return triton_gelu_dispatch