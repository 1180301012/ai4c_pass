"""
Pass: FusedSoftmaxWeightedSum_Gen
Generalized pattern that handles ANY batch size B.

After the framework's dimension_generalization_passes
(naive_call_method_reshape_pass, naive_call_method_view_pass),
concrete batch dimensions (8, 1, 2, ...) become -1.
This single pattern therefore matches all test variants:
  float32/8  (B=8), float16/3 (B=8), float16/9 (B=1),
  float16/0  (B=1), bfloat16/9 (B=1), float16/4 (B=2)
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
    tmp_6 = tmp_5.contiguous()
    return tmp_6


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return fused_softmax_weighted_sum