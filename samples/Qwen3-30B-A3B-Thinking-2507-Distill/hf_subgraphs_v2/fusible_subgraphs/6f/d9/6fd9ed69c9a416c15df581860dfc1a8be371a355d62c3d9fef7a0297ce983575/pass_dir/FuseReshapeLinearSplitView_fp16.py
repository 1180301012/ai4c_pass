"""
Pass: FuseReshapeLinearSplitView_fp16

Matches (fp16/bf16 graphs) reshape(300,-1,256) + F.linear into a single
Triton GEMM. The downstream slice ops stay in the graph.

Uses shared dispatch from shared_dispatch.py so both passes return the SAME
function object, satisfying replacement_func_limit.
"""

import torch
import triton
import triton.language as tl
from pass_dir.shared_dispatch import _dispatch_replacement


def pattern(a, b, c):
    """
    a : bias   [N]     (in_2)
    b : weight [N, K]  (in_3)
    c : input  [1,150,1,512] (in_4) — treated as [M=300, K]
    Returns single linear output [300, 512].
    """
    tmp_9  = c.reshape(300, -1, 256)
    linear = torch.nn.functional.linear(tmp_9, b, a)
    return linear


def replacement_args(a, b, c):
    return (a, b, c, "fp16")


def replacement_func():
    return _dispatch_replacement