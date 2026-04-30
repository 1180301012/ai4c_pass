import torch
import triton
import triton.language as tl
from pass_dir._kernels import dispatch_wrapper


def pattern(in_0, in_1, in_2):
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    return linear


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, "linear")


def replacement_func():
    return dispatch_wrapper