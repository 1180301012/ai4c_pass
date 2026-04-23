import torch
import triton
import triton.language as tl
from pass_dir.shared_replacement import shared_dispatch


def pattern(x):
    tmp_0 = x[slice(None, None, None), slice(None, None, None), None, slice(None, None, None), slice(None, None, None)]
    tmp_1 = tmp_0.expand(1, 1, 8, 3, 256)
    tmp_2 = tmp_1.reshape(1, 8, 3, 256)
    return tmp_2


def replacement_args(x):
    return (x, "broadcast8")


def replacement_func():
    return shared_dispatch