"""
Pass: FuseSiluAvgPoolFlattenDropout_p0_3
Matches: silu -> adaptive_avg_pool2d -> flatten -> dropout(p=0.3, training=False)
Replaces with a single fused Triton kernel.
"""

import torch
from pass_dir.kernel_silu_avgpool import fused_silu_global_avgpool


def pattern(in_0):
    tmp_0 = torch.nn.functional.silu(in_0, inplace=True)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, 1)
    tmp_2 = torch.flatten(tmp_1, 1)
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.3, False, True)
    return tmp_3


def replacement_args(in_0):
    return (in_0,)


def replacement_func():
    return fused_silu_global_avgpool