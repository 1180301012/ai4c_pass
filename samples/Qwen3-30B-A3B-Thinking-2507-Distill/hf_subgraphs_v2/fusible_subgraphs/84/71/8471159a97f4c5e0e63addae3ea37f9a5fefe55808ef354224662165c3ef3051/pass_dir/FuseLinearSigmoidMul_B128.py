"""
Pass: FuseLinearSigmoidMul_B128
Matches the pattern for batch size B=128 (view(128, 64, 1, 1)).
Fuses: linear -> sigmoid -> view -> mul into a single Triton kernel.
"""

import torch
from pass_dir.fused_linear_sigmoid_mul_kernel import fused_linear_sigmoid_mul


def pattern(in_0, in_1, in_2, in_3):
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3  = torch.sigmoid(linear)
    tmp_4  = tmp_3.view(128, 64, 1, 1)
    tmp_5  = in_3 * tmp_4
    return tmp_5


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


def replacement_func():
    return fused_linear_sigmoid_mul