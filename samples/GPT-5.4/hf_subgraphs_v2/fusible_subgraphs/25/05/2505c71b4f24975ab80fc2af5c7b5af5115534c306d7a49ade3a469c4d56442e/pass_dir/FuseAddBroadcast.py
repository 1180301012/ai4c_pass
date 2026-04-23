import torch
from pass_dir.shared_dispatch import replacement_impl


def pattern(a, b):
    out = a + b
    return out


def replacement_args(a, b):
    return (a, b, "add_broadcast")


def replacement_func():
    return replacement_impl