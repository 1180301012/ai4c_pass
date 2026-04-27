"""
Fused pass for batch=1 — high-level ops, no softmax in pattern.
Covers: float16/9, float16/0, and bfloat16/9 graphs.
"""
import torch
from pass_dir.fused_softmax_weighted_sum_impl import fused_weighted_sum


def pattern(in_0, softmax_out):
    tmp_1 = softmax_out.reshape(1, 256)
    tmp_2 = tmp_1.view(1, 256, 1, 1)
    tmp_3 = tmp_2.view(1, 2, 128, 1, 1)
    tmp_4 = tmp_3 * in_0
    tmp_5 = torch.sum(tmp_4, dim=1)
    return tmp_5


def replacement_args(in_0, softmax_out):
    return (in_0, softmax_out)


def replacement_func():
    return fused_weighted_sum