import torch
from pass_dir.shared_dispatch import replacement_impl


def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = torch.nn.functional.relu(in_2, inplace=False)
    tmp_3 = in_1 * tmp_2
    tmp_4 = tmp_3 + in_0
    tmp_5 = torch.nn.functional.max_pool2d(in_3, 2, 1, 0, 1, ceil_mode=True, return_indices=False)
    tmp_6 = torch.cat([tmp_5, tmp_4], dim=1)
    return tmp_6


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3, "whole_fused")


def replacement_func():
    return replacement_impl