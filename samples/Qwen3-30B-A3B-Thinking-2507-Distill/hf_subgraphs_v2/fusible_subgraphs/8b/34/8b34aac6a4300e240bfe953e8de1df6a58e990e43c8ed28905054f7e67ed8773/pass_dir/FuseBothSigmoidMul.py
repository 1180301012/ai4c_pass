"""
Pass: FuseBothSigmoidMul

Matches BOTH sigmoid*mul pairs in one combined pattern and replaces them
with a single Triton kernel call (vs 2 separate calls).

This halves the Python/dispatch overhead, significantly improving small-batch
performance.

Pattern matches:
    chain1: tmp_5 = in_3 * torch.sigmoid(in_4)    [B, C, 64, 64]
    chain2: tmp_7 = in_2 * torch.sigmoid(conv2d)   [B, C, 16, 16]

Both are computed in one kernel launch via combined_sigmoid_mul(a, b, c, d).
"""

import torch
from pass_dir.shared_kernel import combined_sigmoid_mul


def pattern(in_4, in_3, conv2d_out, in_2):
    """
    in_4:      [B, C, 64, 64]  – sigmoid input (result of F.interpolate)
    in_3:      [B, C, 64, 64]  – multiplier
    conv2d_out: [B, C, 16, 16] – sigmoid input (conv2d output)
    in_2:      [B, C, 16, 16]  – multiplier
    """
    tmp_4 = torch.sigmoid(in_4)
    tmp_5 = in_3 * tmp_4
    tmp_6 = torch.sigmoid(conv2d_out)
    tmp_7 = in_2 * tmp_6
    return tmp_5, tmp_7


def replacement_args(in_4, in_3, conv2d_out, in_2):
    return (in_4, in_3, conv2d_out, in_2)


def replacement_func():
    return combined_sigmoid_mul