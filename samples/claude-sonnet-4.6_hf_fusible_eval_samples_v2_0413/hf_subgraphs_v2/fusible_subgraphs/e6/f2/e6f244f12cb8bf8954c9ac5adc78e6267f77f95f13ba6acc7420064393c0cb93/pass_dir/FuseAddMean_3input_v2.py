"""Universal mean pattern (3-input v2 file). Shares fast_mean with all passes."""
import torch
import triton
import triton.language as tl
from pass_dir.shared_kernel import fast_mean


def pattern(x):
    y = x.mean((2, 3), keepdim=True)
    return y


def replacement_args(x):
    return (x,)


def replacement_func():
    return fast_mean