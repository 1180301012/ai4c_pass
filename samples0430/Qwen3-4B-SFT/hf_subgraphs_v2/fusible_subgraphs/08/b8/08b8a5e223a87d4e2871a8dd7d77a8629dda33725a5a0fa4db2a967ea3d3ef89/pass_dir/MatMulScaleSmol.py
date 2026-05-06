"""
Optimization pass: fuse F.linear(x, w, None) + scale * linear_out

Covers SmolLM3-3B pattern:
    linear = torch.nn.functional.linear(in_1, in_0, None)
    tmp_2  = in_2 * linear
    return (tmp_2,)

Uses shared fused_dispatch with route="smol".
"""

import torch
from pass_dir.gemm_scale_kernels import fused_dispatch


def pattern(in_0, in_1, in_2):
    linear = torch.nn.functional.linear(in_1, in_0, None)
    tmp_2  = in_2 * linear
    return tmp_2


def replacement_args(in_0, in_1, in_2):
    # weight, scale, input  →  fused_dispatch(weight, scale, input, "smol")
    return (in_0, in_1, in_2, "smol")


def replacement_func():
    return fused_dispatch