import torch
from pass_dir.shared_kernels import _dispatch


def pattern(in_2, in_5, in_3):
    return torch.cat((in_2, in_5, in_3), dim=2)


def replacement_args(in_2, in_5, in_3):
    return (in_2, in_5, in_3, "cat_dim2")


def replacement_func():
    return _dispatch