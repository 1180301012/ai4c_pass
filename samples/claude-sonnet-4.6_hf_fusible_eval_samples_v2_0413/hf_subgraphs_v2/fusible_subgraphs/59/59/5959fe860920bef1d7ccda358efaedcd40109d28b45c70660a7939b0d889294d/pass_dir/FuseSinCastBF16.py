import torch
from pass_dir.shared_kernels import shared_dispatch


def pattern(x):
    tmp_4 = x.sin()
    tmp_5 = tmp_4 * 1.0
    tmp_7 = tmp_5.to(dtype=torch.bfloat16)
    return tmp_7


def replacement_args(x):
    return (x, "sin_bf16")


def replacement_func():
    return shared_dispatch