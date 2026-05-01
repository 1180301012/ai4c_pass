"""
Pattern: in_0 + in_1; += in_2  →  in_0 + in_1 + in_2
Fuses 3-input element-wise addition with spatial mean into one Triton kernel.

Matched graphs:
  repvgg_d2se.rvgg_in1k_start181_end184_14  (float16 / float32 / bfloat16)
"""

import operator
import torch
from pass_dir.shared_kernels import fused_3input_mean


def pattern(in_0, in_1, in_2):
    tmp_0 = in_0 + in_1
    tmp_0 = operator.iadd(tmp_0, in_2)
    tmp_2 = tmp_0.mean((2, 3), keepdim=True)
    return (tmp_0, tmp_2)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return fused_3input_mean