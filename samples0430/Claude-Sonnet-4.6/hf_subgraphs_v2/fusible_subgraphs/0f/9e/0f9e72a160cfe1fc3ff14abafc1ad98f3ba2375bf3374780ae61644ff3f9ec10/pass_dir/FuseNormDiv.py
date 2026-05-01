import torch
from pass_dir.shared_dispatch import dispatch_wrapper  # noqa: F401


def pattern(x):
    norm = x.norm(p=2, dim=-1, keepdim=True)
    out = x / norm
    return out


def replacement_args(x):
    return (x, "norm_div")


def replacement_func():
    return dispatch_wrapper