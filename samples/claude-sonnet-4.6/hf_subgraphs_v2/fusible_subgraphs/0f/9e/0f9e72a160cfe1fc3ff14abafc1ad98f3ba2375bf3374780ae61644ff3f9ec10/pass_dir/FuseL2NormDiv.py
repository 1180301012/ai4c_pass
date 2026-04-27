import torch
from pass_dir.shared_kernels import shared_dispatch


def pattern(x):
    norm = x.norm(p=2, dim=-1, keepdim=True)
    out = x / norm
    return out


def replacement_args(x):
    return (x, "l2norm")


def replacement_func():
    return shared_dispatch