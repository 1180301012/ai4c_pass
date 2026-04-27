import torch
import triton
import triton.language as tl
from pass_dir.shared_kernels import shared_dispatch


def pattern(x, linear_w, linear_b):
    x_slice = x[(slice(None, None, None), 0)]
    lin = torch.nn.functional.linear(x_slice, linear_w, linear_b)
    tanh_out = torch.tanh(lin)
    return tanh_out


def replacement_args(x, linear_w, linear_b):
    return (x, linear_w, linear_b, "elim")


def replacement_func():
    return shared_dispatch