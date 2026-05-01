"""
Pass: FuseDropoutLinear
Matches the bigbird pattern:
    tmp_3 = torch.nn.functional.dropout(in_2, 0.1, False, False)
    linear = torch.nn.functional.linear(tmp_3, in_1, in_0)
Dropout is a no-op (training=False), so this reduces to a plain linear layer.
Works for both float16 and bfloat16 inputs.
"""
import torch
from pass_dir.shared_linear import fused_linear


def pattern(in_0, in_1, in_2):
    tmp_3 = torch.nn.functional.dropout(in_2, 0.1, False, False)
    linear = torch.nn.functional.linear(tmp_3, in_1, in_0)
    return linear


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return fused_linear