"""
Pass: FuseLinearSplitView_fp32

Matches (float32 graph) the F.linear(in_5, in_1, in_0) call
(2D input, single output). The downstream slice+view ops stay in the graph.

Uses shared dispatch from shared_dispatch.py so both passes return the SAME
function object, satisfying replacement_func_limit.
"""

import torch
import triton
import triton.language as tl
from pass_dir.shared_dispatch import _dispatch_replacement


def pattern(a, b, c):
    """
    a : bias   [N]     (in_0)
    b : weight [N, K]  (in_1)
    c : input  [M, K]  (in_5)
    Returns single linear output [M, N].
    """
    return torch.nn.functional.linear(c, b, a)


def replacement_args(a, b, c):
    return (a, b, c, "fp32")


def replacement_func():
    return _dispatch_replacement