"""
Pass: FusedSoftmaxWeightedSum_B8_ATen
ATen-level pattern for B=8.
Uses torch.ops.aten.* ops matching what torch.export/_decomposed graphs contain.
"""

import torch
from pass_dir.fused_softmax_weighted_sum_kernel import fused_softmax_weighted_sum


def pattern(in_0, in_1):
    tmp_0 = torch.ops.aten._softmax.default(in_1, 1, False)
    tmp_1 = torch.ops.aten.reshape.default(tmp_0, [8, -1])
    tmp_2 = torch.ops.aten.view.default(tmp_1, [8, -1, 1, 1])
    tmp_3 = torch.ops.aten.view.default(tmp_2, [8, 2, -1, 1, 1])
    tmp_4 = torch.ops.aten.mul.Tensor(tmp_3, in_0)
    tmp_5 = torch.ops.aten.sum.dim_IntList(tmp_4, [1], False)
    return tmp_5


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return fused_softmax_weighted_sum