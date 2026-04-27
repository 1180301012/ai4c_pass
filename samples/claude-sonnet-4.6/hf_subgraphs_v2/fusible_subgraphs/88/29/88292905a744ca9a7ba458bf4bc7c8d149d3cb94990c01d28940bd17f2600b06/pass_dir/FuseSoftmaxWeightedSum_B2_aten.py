"""
Fused pass for batch=2 — aten-level ops, no softmax in pattern.
Uses torch.ops.aten.* to match the decomposed/exported graph.
Covers: float16/4 graph.
"""
import torch
from pass_dir.fused_softmax_weighted_sum_impl import fused_weighted_sum


def pattern(in_0, softmax_out):
    tmp_1 = torch.ops.aten.view.default(softmax_out, [2, 256])
    tmp_2 = torch.ops.aten.view.default(tmp_1, [2, 256, 1, 1])
    tmp_3 = torch.ops.aten.view.default(tmp_2, [2, 2, 128, 1, 1])
    tmp_4 = torch.ops.aten.mul.Tensor(tmp_3, in_0)
    tmp_5 = torch.ops.aten.sum.dim_IntList(tmp_4, [1], False)
    return tmp_5


def replacement_args(in_0, softmax_out):
    return (in_0, softmax_out)


def replacement_func():
    return fused_weighted_sum