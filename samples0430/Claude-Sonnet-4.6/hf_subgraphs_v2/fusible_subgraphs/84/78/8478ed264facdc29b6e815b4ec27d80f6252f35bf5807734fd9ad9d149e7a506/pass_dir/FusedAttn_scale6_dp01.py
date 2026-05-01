"""
Fused multi-head attention pass for scale = 6.0, dropout_p = 0.1 (training=False).
Covers hf-tiny-model-private/tiny-random-MobileViTModel (float16 variant).
"""

import torch
from pass_dir.fused_attn_core import fused_attn

_SCALE = 6.0


def pattern(in_0, in_1, in_2):
    matmul = torch.matmul(in_0, in_1)
    tmp_1 = matmul / _SCALE
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.1, False, False)
    matmul_1 = torch.matmul(tmp_3, in_2)
    tmp_5 = matmul_1.permute(0, 2, 1, 3)
    tmp_6 = tmp_5.contiguous()
    return tmp_6


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, _SCALE)


def replacement_func():
    return fused_attn