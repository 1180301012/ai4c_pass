"""
Pass: Spatial mean over dims (2, 3) with keepdim=True via Triton.
Pattern matches: x.mean((2, 3), keepdim=True)
Single-output pattern — no multi-output FX replace_pattern issues.
"""
import torch
from pass_dir.dw_conv_mean_kernel import dispatch_mean


def pattern(in_1):
    return in_1.mean((2, 3), keepdim=True)


def replacement_args(in_1):
    return (in_1,)


def replacement_func():
    return dispatch_mean