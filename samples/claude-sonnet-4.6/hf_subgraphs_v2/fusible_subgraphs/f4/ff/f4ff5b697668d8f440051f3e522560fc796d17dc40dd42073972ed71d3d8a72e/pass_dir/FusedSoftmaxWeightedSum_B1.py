"""
Fused pass for B=1: softmax + weighted coordinate sum.
Matches: float16/9, bfloat16/9  (in_2 shape [1, 17, 4096])
"""

import torch
from pass_dir.kernel_impl import fused_softmax_weighted_sum


def pattern(in_0, in_1, in_2):
    tmp_2 = torch.nn.functional.softmax(in_2, dim=2)
    tmp_3 = tmp_2.reshape(-1, 17, 64, 64)
    tmp_4 = tmp_3.mul(in_0)
    tmp_5 = tmp_4.reshape(1, 17, -1)
    tmp_6 = torch.sum(tmp_5, dim=2, keepdim=True)
    tmp_7 = tmp_3.mul(in_1)
    tmp_8 = tmp_7.reshape(1, 17, -1)
    tmp_9 = torch.sum(tmp_8, dim=2, keepdim=True)
    tmp_10 = torch.cat([tmp_6, tmp_9], dim=-1)
    return (tmp_3, tmp_10)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return fused_softmax_weighted_sum