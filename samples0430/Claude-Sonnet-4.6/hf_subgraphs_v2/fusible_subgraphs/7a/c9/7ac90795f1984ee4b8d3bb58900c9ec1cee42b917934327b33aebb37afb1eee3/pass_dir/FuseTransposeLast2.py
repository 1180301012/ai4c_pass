import torch
from pass_dir.shared_kernels import triton_transpose_last2


def pattern(in_2):
    return in_2.transpose(-1, -2)


def replacement_args(in_2):
    return (in_2,)


def replacement_func():
    return triton_transpose_last2