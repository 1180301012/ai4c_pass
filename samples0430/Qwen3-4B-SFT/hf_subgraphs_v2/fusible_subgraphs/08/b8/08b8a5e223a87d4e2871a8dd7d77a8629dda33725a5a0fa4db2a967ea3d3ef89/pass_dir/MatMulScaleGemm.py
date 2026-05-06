"""
Optimization pass: fuse F.linear(x, w, None) + scale * linear_out

Covers gemma-1.1-2b-it pattern:
    linear = torch.nn.functional.linear(in_2, in_0, None)
    tmp_2  = in_1 * linear
    return (tmp_2,)

Uses shared fused_dispatch with route="gemm".
"""

import torch
from pass_dir.gemm_scale_kernels import fused_dispatch


def pattern(in_0, in_1, in_2):
    linear = torch.nn.functional.linear(in_2, in_0, None)
    tmp_2  = in_1 * linear
    return tmp_2


def replacement_args(in_0, in_1, in_2):
    # weight, scale, input  →  fused_dispatch(weight, scale, input, "gemm")
    return (in_0, in_1, in_2, "gemm")


def replacement_func():
    return fused_dispatch