import torch
import triton
import triton.language as tl

from pass_dir.shared_kernels import shared_replacement_dispatch


def pattern(in_3):
    tmp_5 = in_3.sum(dim=3, keepdim=True)
    tmp_6 = in_3 / tmp_5
    return tmp_6


def replacement_args(in_3):
    return (in_3, "row_norm_dim3_keepdim")


def replacement_func():
    return shared_replacement_dispatch