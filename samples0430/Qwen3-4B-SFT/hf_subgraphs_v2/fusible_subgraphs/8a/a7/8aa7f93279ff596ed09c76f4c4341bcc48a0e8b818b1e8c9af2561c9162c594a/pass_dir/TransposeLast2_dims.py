import torch
import triton
import triton.language as tl
from pass_dir.shared_dispatch import shared_dispatch


def pattern(x):
    result = x.transpose(-2, -1)
    return result


def replacement_args(x):
    return (x, "transpose")


def replacement_func():
    return shared_dispatch