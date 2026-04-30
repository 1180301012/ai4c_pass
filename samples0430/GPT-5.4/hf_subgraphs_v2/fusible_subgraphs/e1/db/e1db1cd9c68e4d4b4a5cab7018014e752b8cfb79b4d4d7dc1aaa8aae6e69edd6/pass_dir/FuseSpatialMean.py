import torch
import triton
import triton.language as tl
from pass_dir.shared_fused_kernels import shared_replacement_func


def pattern(in_0):
    tmp_0 = in_0.mean((2, 3), keepdim=True)
    return tmp_0


def replacement_args(in_0):
    return (in_0, 'mean')


def replacement_func():
    return shared_replacement_func()