"""
Pass: FuseCastLinear_bfloat16
Matches the RECT_L bfloat16 graph after dropout (p=0.0, training=False) is
folded away by the compiler, leaving just:
    to     = in_2.to(torch.bfloat16)
    linear = torch.nn.functional.linear(to, in_1, in_0)
"""
import torch
from pass_dir.shared_linear import fused_linear


def pattern(in_0, in_1, in_2):
    to = in_2.to(torch.bfloat16)
    linear = torch.nn.functional.linear(to, in_1, in_0)
    return linear


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return fused_linear