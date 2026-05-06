import torch
import triton
import triton.language as tl
from pass_dir.shared_dispatch import shared_dispatch


def pattern(x):
    result = x * 0.1767766952966369
    return result


def replacement_args(x):
    return (x, "mul")


def replacement_func():
    return shared_dispatch