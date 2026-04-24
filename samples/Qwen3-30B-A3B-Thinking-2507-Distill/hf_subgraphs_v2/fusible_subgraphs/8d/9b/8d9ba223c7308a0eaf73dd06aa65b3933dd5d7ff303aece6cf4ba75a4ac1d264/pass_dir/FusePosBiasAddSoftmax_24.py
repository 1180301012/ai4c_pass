"""
AI4C pass: FusePosBiasAddSoftmax_24

Matches the FULL model for H=24 heads (Graphs: bfloat16/float16/9/start884_end905_24)

Replacement: Triton GEMM + fused gather+sigmoid+add+softmax kernel.
"""

import torch
from pass_dir.pos_bias_kernel import _run_fused_24


def pattern(in_4, in_1, in_0, in_2, in_3):
    linear = torch.nn.functional.linear(in_4, in_1, None)
    tmp_3 = linear.view(-1, 24)
    tmp_4 = in_0.view(-1)
    tmp_5 = tmp_3[tmp_4]
    tmp_6 = tmp_5.view(64, 64, -1)
    tmp_7 = tmp_6.permute(2, 0, 1)
    tmp_8 = tmp_7.contiguous()
    tmp_9 = torch.sigmoid(tmp_8)
    tmp_10 = 16 * tmp_9
    tmp_11 = tmp_10.unsqueeze(0)
    tmp_12 = in_2 + tmp_11
    tmp_13 = tmp_12.view(1, 16, 24, 64, 64)
    tmp_14 = in_3.unsqueeze(1)
    tmp_15 = tmp_14.unsqueeze(0)
    tmp_16 = tmp_13 + tmp_15
    tmp_17 = in_3.unsqueeze(1)
    tmp_18 = tmp_17.unsqueeze(0)
    tmp_19 = tmp_16 + tmp_18
    tmp_20 = tmp_19.view(-1, 24, 64, 64)
    tmp_21 = torch.nn.functional.softmax(tmp_20, dim=-1)
    tmp_22 = torch.nn.functional.dropout(tmp_21, 0.0, False, False)
    return (tmp_22,)


def replacement_args(in_4, in_1, in_0, in_2, in_3):
    return (in_4, in_1, in_0, in_2, in_3)


def replacement_func():
    return _run_fused_24