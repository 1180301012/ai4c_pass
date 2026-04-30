import torch
import triton
import triton.language as tl
from pass_dir._kernels import dispatch_wrapper


def pattern(in_3):
    mean = in_3.mean(-2)
    return mean


def replacement_args(in_3):
    return (in_3, "mean")


def replacement_func():
    return dispatch_wrapper