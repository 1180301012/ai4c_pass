"""
Pass: FusedSoftmaxWeightedSum_B1
Fuses softmax + reshape/view chain + elementwise multiply + sum
for batch-size B=1.

Matches the graph:
  softmax(in_1, dim=1)                    # [1, 2, 1, C]
  .reshape(1, -1)                         # [1, 2*C]
  .view(1, -1, 1, 1)                      # [1, 2*C, 1, 1]
  .view(1, 2, -1, 1, 1)                   # [1, 2, C, 1, 1]
  * in_0                                  # [1, 2, C, H, W]  (broadcast)
  .sum(dim=1)                             # [1, C, H, W]
  .contiguous()
"""

import torch
from pass_dir.fused_softmax_weighted_sum_kernel import fused_softmax_weighted_sum


def pattern(in_0, in_1):
    tmp_0 = torch.nn.functional.softmax(in_1, dim=1)
    tmp_1 = tmp_0.reshape(1, -1)
    tmp_2 = tmp_1.view(1, -1, 1, 1)
    tmp_3 = tmp_2.view(1, 2, -1, 1, 1)
    tmp_4 = tmp_3 * in_0
    tmp_5 = torch.sum(tmp_4, dim=1)
    tmp_6 = tmp_5.contiguous()
    return tmp_6


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return fused_softmax_weighted_sum