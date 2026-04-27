"""
Pass: FuseGeluAddLayerNorm_C32_H64_W48

Single-return pattern: gelu(in_2[1,32,64,48]) → flatten → transpose
  → contiguous → add in_3[1,3072,32] → permute/view/view/permute → tmp_10.
"""

import torch
from pass_dir.shared_dispatch import fused_dispatch


def pattern(in_2, in_3):
    tmp_2  = torch.nn.functional.gelu(in_2, approximate='none')
    tmp_3  = tmp_2.flatten(2)
    tmp_4  = tmp_3.transpose(1, 2)
    tmp_5  = tmp_4.contiguous()
    tmp_6  = in_3 + tmp_5
    tmp_7  = tmp_6.permute(0, 2, 1)
    tmp_8  = tmp_7.view(1, 32, 64, 48)
    tmp_9  = tmp_8.view(1, 32, -1)
    tmp_10 = tmp_9.permute(0, 2, 1)
    return tmp_10


def replacement_args(in_2, in_3):
    return (in_2, in_3, "C32_H64_W48")


def replacement_func():
    return fused_dispatch