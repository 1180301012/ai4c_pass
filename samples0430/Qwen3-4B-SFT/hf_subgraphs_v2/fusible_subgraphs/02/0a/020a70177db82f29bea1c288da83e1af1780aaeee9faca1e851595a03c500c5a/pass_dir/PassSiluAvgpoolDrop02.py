"""
Pass: Fuse SiLU + GlobalAvgPool + Flatten + Dropout(p=0.2) into one Triton kernel.
Matches models with dropout probability 0.2 (e.g. efficientnet_b0).
"""

import torch
from pass_dir.silu_avgpool_shared import dispatch_silu_avgpool


def pattern(in_0):
    tmp_0 = torch.nn.functional.silu(in_0, inplace=True)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, 1)
    tmp_2 = torch.flatten(tmp_1, 1)
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.2, False, True)
    return tmp_3


def replacement_args(in_0):
    return (in_0, "drop02")


def replacement_func():
    return dispatch_silu_avgpool