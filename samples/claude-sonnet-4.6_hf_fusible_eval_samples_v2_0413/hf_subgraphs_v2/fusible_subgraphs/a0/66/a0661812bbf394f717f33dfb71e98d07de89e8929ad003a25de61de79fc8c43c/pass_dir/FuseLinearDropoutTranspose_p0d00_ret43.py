"""
Pass: FuseLinearDropoutTranspose_p0d00_ret43
Matches: linear -> dropout(p=0.0, training=False) -> transpose(1,2)
Returns: (transposed_out, dropout_out)   [positions tmp_4, tmp_3]
"""
import torch
from pass_dir.linear_transpose_shared import linear_transpose_ret43


def pattern(in_0, in_1, in_2):
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3  = torch.nn.functional.dropout(linear, 0.0, False, False)
    tmp_4  = tmp_3.transpose(1, 2)
    return tmp_4, tmp_3


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return linear_transpose_ret43