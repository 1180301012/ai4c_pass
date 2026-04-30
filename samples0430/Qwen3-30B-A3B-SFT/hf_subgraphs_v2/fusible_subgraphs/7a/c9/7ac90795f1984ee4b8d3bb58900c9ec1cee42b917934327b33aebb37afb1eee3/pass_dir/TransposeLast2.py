"""
Pass: replace in_2.transpose(-1, -2) with a Triton tile-transpose kernel.
  Generic across all ConvBERT graphs (4-D tensors, last two dims swapped).
"""
import torch
from pass_dir.shared_kernels import shared_dispatch


def pattern(x):
    tmp_2 = x.transpose(-1, -2)
    return tmp_2


def replacement_args(x):
    # a=x, b=x (dummy matmul args), c=x (actual tensor), route selects branch
    return (x, x, x, "transpose")


def replacement_func():
    return shared_dispatch