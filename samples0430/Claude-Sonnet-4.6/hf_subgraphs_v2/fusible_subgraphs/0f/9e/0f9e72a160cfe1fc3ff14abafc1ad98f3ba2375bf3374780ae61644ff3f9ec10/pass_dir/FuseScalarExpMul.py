import torch
from pass_dir.shared_dispatch import dispatch_wrapper  # noqa: F401


def pattern(s, x):
    e = s.exp()
    out = e * x
    return out


def replacement_args(s, x):
    return (s, x, "exp_mul")


def replacement_func():
    return dispatch_wrapper