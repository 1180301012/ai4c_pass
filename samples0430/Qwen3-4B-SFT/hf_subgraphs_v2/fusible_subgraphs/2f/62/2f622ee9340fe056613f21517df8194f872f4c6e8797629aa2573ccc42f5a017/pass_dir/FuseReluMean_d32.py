"""
Pass: Fuse relu (inplace) + spatial mean into a single Triton kernel.
Matches the pattern in resnest50d subgraph uses divisor = 32 (in_0 // 32).
"""

import torch
from pass_dir._relu_mean_shared import dispatch_fused_relu_mean


def pattern(in_0, in_1):
    tmp_0 = torch.nn.functional.relu(in_1, inplace=True)
    tmp_3 = tmp_0.mean((2, 3), keepdim=True)
    return (tmp_0, tmp_3)


def replacement_args(in_0, in_1):
    # in_0 (sym_sum scalar) is not part of the fused tensor computation.
    # We pass it so the graph structure matches, but the dispatch ignores it.
    return (in_1, "d32")


def replacement_func():
    return dispatch_fused_relu_mean