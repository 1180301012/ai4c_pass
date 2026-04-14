import torch
from pass_dir.shared_kernels import shared_dispatch


def pattern(x):
    tmp_2 = x.cos()
    tmp_3 = tmp_2 * 1.0
    tmp_6 = tmp_3.to(dtype=torch.float32)
    return tmp_6


def replacement_args(x):
    return (x, "cos_f32")


def replacement_func():
    return shared_dispatch