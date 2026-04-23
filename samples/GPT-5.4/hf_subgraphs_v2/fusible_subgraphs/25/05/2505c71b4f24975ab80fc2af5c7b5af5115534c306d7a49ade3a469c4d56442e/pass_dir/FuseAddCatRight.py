import torch
from pass_dir.shared_dispatch import replacement_impl


def pattern(a, b, c):
    tmp = a + b
    out = torch.cat([c, tmp], dim=1)
    return out


def replacement_args(a, b, c):
    return (a, b, c, "add_cat_right")


def replacement_func():
    return replacement_impl