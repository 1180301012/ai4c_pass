import torch
from pass_dir.shared_dispatch import dispatch


def pattern(x):
    return x * 0.1767766952966369


def replacement_args(x):
    return (x, "mul")


def replacement_func():
    return dispatch