"""
Pass: Replace x.mean((2, 3), keepdim=True) with a Triton spatial-mean kernel.
Single-input / single-output pattern — avoids multi-output replacement issues.
"""
import torch
from pass_dir.dw_conv_mean_kernel import fast_spatial_mean_2d


def pattern(x):
    return x.mean((2, 3), keepdim=True)


def replacement_args(x):
    return (x,)


def replacement_func():
    return fast_spatial_mean_2d