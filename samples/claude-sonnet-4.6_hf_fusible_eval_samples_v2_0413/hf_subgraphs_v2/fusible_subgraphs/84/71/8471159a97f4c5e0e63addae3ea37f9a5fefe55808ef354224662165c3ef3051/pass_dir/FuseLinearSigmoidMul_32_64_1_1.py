"""
Pass: FuseLinearSigmoidMul_32_64_1_1
Matches: linear -> sigmoid -> view(32,64,1,1) -> multiply
Targets: float16/3 (in_3=[32,64,56,56])
"""

import torch
import triton
import triton.language as tl
from pass_dir.fused_lsm_kernel import fused_linear_sigmoid_mul


def pattern(in_0, in_1, in_2, in_3):
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = torch.sigmoid(linear)
    tmp_4 = tmp_3.view(32, 64, 1, 1)
    tmp_5 = in_3 * tmp_4
    return tmp_5


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


def replacement_func():
    return fused_linear_sigmoid_mul