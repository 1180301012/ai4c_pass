import torch
import triton
import triton.language as tl

from pass_dir.shared_scale_then_transpose import fused_scale_then_transpose_dispatch


def pattern(in_0):
    tmp_0 = in_0 / 1.6817928305074292
    tmp_1 = tmp_0.transpose(-1, -2)
    return tmp_1


def replacement_args(in_0):
    return (in_0, "div_16817928305074292")


def replacement_func():
    return fused_scale_then_transpose_dispatch