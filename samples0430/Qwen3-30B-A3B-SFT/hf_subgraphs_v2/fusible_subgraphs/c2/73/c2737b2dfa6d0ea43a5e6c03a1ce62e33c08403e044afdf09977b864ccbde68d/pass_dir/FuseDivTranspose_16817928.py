import torch
from pass_dir.shared_kernel import dispatch_fused_div_transpose


def pattern(x):
    tmp_0 = x / 1.6817928305074292
    tmp_1 = tmp_0.transpose(-1, -2)
    return tmp_1


def replacement_args(x):
    return (x, "r16817")


def replacement_func():
    return dispatch_fused_div_transpose