import torch
from pass_dir.shared_dispatch import replacement_impl


def pattern(a, b):
    out = torch.cat([a, b], dim=1)
    return out


def replacement_args(a, b):
    return (a, b, "cat_dim1")


def replacement_func():
    return replacement_impl