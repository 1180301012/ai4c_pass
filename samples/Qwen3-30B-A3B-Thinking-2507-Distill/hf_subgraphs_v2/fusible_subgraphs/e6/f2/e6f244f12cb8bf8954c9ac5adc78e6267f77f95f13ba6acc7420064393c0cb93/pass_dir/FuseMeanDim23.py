"""
Pass: FuseMeanDim23
Matches: x.mean((2, 3), keepdim=True)
Applies a high-performance Triton mean reduction over spatial dims (H, W).
"""
import torch
from pass_dir._add_mean_kernel import triton_mean_dim23


def pattern(x):
    return x.mean((2, 3), keepdim=True)


def replacement_args(x):
    return (x,)


def replacement_func():
    return triton_mean_dim23