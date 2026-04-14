"""
SmolLM3 pattern: linear(in_1, in_0, None) followed by in_2 * linear
Uses the shared dispatch_all function (route="smollm").
"""

import torch
from pass_dir.shared_kernels import dispatch_all


def pattern(in_0, in_1, in_2):
    linear = torch.nn.functional.linear(in_1, in_0, None)
    tmp_2  = in_2 * linear
    return tmp_2


def replacement_args(in_0, in_1, in_2):
    # a0=weight(in_0), a1=input(in_1), a2=gate(in_2), route="smollm"
    return (in_0, in_1, in_2, "smollm")


def replacement_func():
    return dispatch_all