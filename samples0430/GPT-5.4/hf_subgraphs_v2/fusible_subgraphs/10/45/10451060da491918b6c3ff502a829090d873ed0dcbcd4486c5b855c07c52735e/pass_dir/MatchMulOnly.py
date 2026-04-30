import torch
from pass_dir.shared_dispatch import replacement_func


def pattern(a, b):
    out = a * b
    return out


def replacement_args(a, b):
    return (a, b, 'mul')