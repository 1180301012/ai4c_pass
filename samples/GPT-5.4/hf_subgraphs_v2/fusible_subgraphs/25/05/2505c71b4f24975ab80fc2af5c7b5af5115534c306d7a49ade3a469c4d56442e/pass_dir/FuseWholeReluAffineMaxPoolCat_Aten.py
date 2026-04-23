import operator
import torch
from pass_dir.shared_dispatch import replacement_impl


def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = torch.ops.aten.relu.default(in_2)
    tmp_3 = torch.ops.aten.mul.Tensor(in_1, tmp_2)
    tmp_4 = torch.ops.aten.add.Tensor(tmp_3, in_0)
    tmp_5_with_idx = torch.ops.aten.max_pool2d_with_indices.default(in_3, [2, 2], [1, 1], [0, 0], [1, 1], True)
    tmp_5 = operator.getitem(tmp_5_with_idx, 0)
    tmp_6 = torch.ops.aten.cat.default([tmp_5, tmp_4], 1)
    return tmp_6


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3, "whole_fused")


def replacement_func():
    return replacement_impl