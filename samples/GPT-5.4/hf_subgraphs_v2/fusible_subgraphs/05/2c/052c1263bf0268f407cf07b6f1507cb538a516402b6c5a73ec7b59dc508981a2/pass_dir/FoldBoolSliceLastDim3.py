import torch
import triton
import triton.language as tl
from pass_dir.shared_replacement import shared_dispatch


def pattern(x):
    tmp_0 = x[slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 3, None)]
    return tmp_0


def replacement_args(x):
    return (x, "identity")


def replacement_func():
    return shared_dispatch