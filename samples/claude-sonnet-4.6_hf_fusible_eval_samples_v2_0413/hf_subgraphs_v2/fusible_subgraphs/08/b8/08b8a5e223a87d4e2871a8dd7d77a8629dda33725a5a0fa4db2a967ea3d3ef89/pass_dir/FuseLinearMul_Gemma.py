"""
Gemma pattern: linear(in_2, in_0, None) followed by in_1 * linear
Uses the shared dispatch_all function (route="gemma").
"""

import torch
from pass_dir.shared_kernels import dispatch_all


def pattern(in_0, in_1, in_2):
    linear = torch.nn.functional.linear(in_2, in_0, None)
    tmp_2  = in_1 * linear
    return tmp_2


def replacement_args(in_0, in_1, in_2):
    # a0=weight(in_0), a1=gate(in_1), a2=input(in_2), route="gemma"
    return (in_0, in_1, in_2, "gemma")


def replacement_func():
    return dispatch_all