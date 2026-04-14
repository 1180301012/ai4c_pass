"""
Pass: FuseLinearDropoutTranspose_p0d00_ret34
Matches: linear(in_2, in_1, in_0) -> dropout(p=0.0, training=False)
Returns single fused output (transpose view remains in graph).
Covers ALL p=0.0 graphs regardless of return order (ret34 and ret43).
"""
import torch
from pass_dir.linear_transpose_shared import fused_linear


def pattern(in_0, in_1, in_2):
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    out = torch.nn.functional.dropout(linear, 0.0, False, False)
    return out


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return fused_linear