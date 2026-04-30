"""
Pass: FusedExpMul
Matches in_0.exp() * in_2  →  single output (tmp_6).
Uses the shared dispatch_wrapper (route="exp_mul") so replacement_func_limit is satisfied.
"""
import torch
import triton
import triton.language as tl
from pass_dir.shared_kernels import dispatch_wrapper


def pattern(in_0, in_2):
    tmp_5 = in_0.exp()
    tmp_6 = tmp_5 * in_2
    return tmp_6


def replacement_args(in_0, in_2):
    return (in_0, in_2, "exp_mul")


def replacement_func():
    return dispatch_wrapper