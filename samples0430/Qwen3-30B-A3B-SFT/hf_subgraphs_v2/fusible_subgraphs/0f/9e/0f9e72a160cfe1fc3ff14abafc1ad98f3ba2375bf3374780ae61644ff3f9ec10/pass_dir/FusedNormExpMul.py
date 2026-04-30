"""
Pass: FusedNormExpMul
Matches in_1.norm(p=2,dim=-1,keepdim=True) / in_1  →  single output (tmp_2 or tmp_4).
Uses the shared dispatch_wrapper (route="norm_div") so replacement_func_limit is satisfied.
"""
import torch
import triton
import triton.language as tl
from pass_dir.shared_kernels import dispatch_wrapper


def pattern(in_1):
    tmp_1 = in_1.norm(p=2, dim=-1, keepdim=True)
    tmp_2 = in_1 / tmp_1
    return tmp_2


def replacement_args(in_1):
    return (in_1, "norm_div")


def replacement_func():
    return dispatch_wrapper