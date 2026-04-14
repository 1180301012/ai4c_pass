"""
Pass: FuseLinearSigmoidMul_1_64_1_1
Matches: linear -> sigmoid -> view(1,64,1,1) -> multiply
Targets: bfloat16/1 (in_3=[1,64,64,64]) and bfloat16/2 (in_3=[1,64,96,96])
"""

import torch
import triton
import triton.language as tl
from pass_dir.fused_lsm_kernel import fused_linear_sigmoid_mul


def pattern(in_0, in_1, in_2, in_3):
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = torch.sigmoid(linear)
    tmp_4 = tmp_3.view(1, 64, 1, 1)
    tmp_5 = in_3 * tmp_4
    return tmp_5


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


def replacement_func():
    return fused_linear_sigmoid_mul