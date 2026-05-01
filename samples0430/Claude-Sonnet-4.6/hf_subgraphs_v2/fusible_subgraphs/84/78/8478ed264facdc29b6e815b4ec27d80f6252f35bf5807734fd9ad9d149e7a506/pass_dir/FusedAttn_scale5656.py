"""
Fused multi-head attention pass for scale = 5.656854249492381 (= sqrt(32)).
Covers nvidia/mit-b0 and similar models.
"""

import torch
from pass_dir.fused_attn_core import fused_attn

_SCALE = 5.656854249492381


def pattern(in_0, in_1, in_2):
    matmul = torch.matmul(in_0, in_1)
    tmp_1 = matmul / _SCALE
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.0, False, False)
    matmul_1 = torch.matmul(tmp_3, in_2)
    tmp_5 = matmul_1.permute(0, 2, 1, 3)
    tmp_6 = tmp_5.contiguous()
    return tmp_6


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, _SCALE)


def replacement_func():
    return fused_attn