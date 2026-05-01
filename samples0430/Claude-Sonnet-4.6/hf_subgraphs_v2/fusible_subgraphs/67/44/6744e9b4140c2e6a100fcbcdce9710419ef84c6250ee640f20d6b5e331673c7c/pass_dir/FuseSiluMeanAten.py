"""Aten-level fallback: matches aten.mean.dim only.

Used when the compiled graph is at the aten level. Shares fast_spatial_mean
with FuseSiluMeanView.py to satisfy output_pass_replacement_func_limit=1.
"""
import torch
import triton
import triton.language as tl
from pass_dir.FuseSiluMeanView import fast_spatial_mean


def pattern(in_1):
    return torch.ops.aten.mean.dim(in_1, [2, 3], False)


def replacement_args(in_1):
    return (in_1,)


def replacement_func():
    return fast_spatial_mean