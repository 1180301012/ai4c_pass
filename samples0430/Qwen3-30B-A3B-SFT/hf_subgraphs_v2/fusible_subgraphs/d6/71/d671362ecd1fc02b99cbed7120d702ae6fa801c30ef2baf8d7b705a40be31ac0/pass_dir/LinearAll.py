"""
Single-output pass: replace torch.nn.functional.linear(x, W, b) with a
Triton GEMM + bias kernel.  Works for all batch sizes (B=1,4,8,128,256,512)
since the pattern contains no reshape with a hardcoded B value.

Downstream reshape/split/permute/transposes remain in the graph (they are free
view operations).
"""

import torch
from pass_dir.linear_triton import triton_linear_bias


def pattern(in_2, in_1, x):
    """Match:  linear(x, weight=in_2, bias=in_1)"""
    return torch.nn.functional.linear(x, in_2, in_1)


def replacement_args(in_2, in_1, x):
    return (in_2, in_1, x)


def replacement_func():
    return triton_linear_bias