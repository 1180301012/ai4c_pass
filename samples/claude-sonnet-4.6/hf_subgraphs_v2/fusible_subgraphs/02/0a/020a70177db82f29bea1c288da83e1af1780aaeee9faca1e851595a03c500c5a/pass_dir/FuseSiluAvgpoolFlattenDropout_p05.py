"""
Fuse: silu(inplace=True) -> adaptive_avg_pool2d(1) -> flatten(1) -> dropout(p=0.5, training=False)
Dropout with training=False is identity, so the effective fusion is:
  silu + global_avg_pool + flatten  →  single Triton kernel
"""

import torch
from pass_dir.silu_avgpool_kernel import silu_avgpool_fused


def pattern(in_0):
    tmp_0 = torch.nn.functional.silu(in_0, inplace=True)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, 1)
    tmp_2 = torch.flatten(tmp_1, 1)
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.5, False, True)
    return (tmp_3,)


def replacement_args(in_0):
    return (in_0,)


def replacement_func():
    return silu_avgpool_fused