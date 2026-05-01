"""
Pattern: 0 + in_1; += in_0  →  effectively in_0 + in_1
Fuses 2-input element-wise addition with spatial mean into one Triton kernel.

Matched graphs:
  mobileone_s4.apple_in1k_start228_end231_3  (float16 / float32 / bfloat16)
"""

import operator
import torch
from pass_dir.shared_kernels import fused_2input_mean


def pattern(in_0, in_1):
    tmp_0 = 0 + in_1
    tmp_0 = operator.iadd(tmp_0, in_0)
    tmp_2 = tmp_0.mean((2, 3), keepdim=True)
    return (tmp_0, tmp_2)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return fused_2input_mean