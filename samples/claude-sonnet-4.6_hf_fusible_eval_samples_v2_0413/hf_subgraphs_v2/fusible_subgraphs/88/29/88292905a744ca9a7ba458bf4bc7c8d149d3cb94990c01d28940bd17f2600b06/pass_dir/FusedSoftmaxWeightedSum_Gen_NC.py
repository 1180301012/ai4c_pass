"""
Pass: FusedSoftmaxWeightedSum_Gen_NC
Generalized pattern (no contiguous) - handles ANY batch size B.
Variant without the trailing .contiguous() for graphs where it is elided.
"""

import torch
from pass_dir.fused_softmax_weighted_sum_kernel import fused_softmax_weighted_sum


def pattern(in_0, in_1):
    tmp_0 = torch.nn.functional.softmax(in_1, dim=1)
    tmp_1 = tmp_0.reshape(-1, -1)
    tmp_2 = tmp_1.view(-1, -1, 1, 1)
    tmp_3 = tmp_2.view(-1, 2, -1, 1, 1)
    tmp_4 = tmp_3 * in_0
    tmp_5 = torch.sum(tmp_4, dim=1)
    return tmp_5


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return fused_softmax_weighted_sum