"""
RTMPose standalone GEMM pass: matches linear(in_3, in_0, None) where the
output is directly returned (observable).  The multiply is handled separately
by LinearAndBroadcastMul_RTMPose.

Uses the shared dispatch_all function (route="rtmpose_gemm").
"""

import torch
from pass_dir.shared_kernels import dispatch_all


def pattern(in_0, in_3):
    linear = torch.nn.functional.linear(in_3, in_0, None)
    return linear


def replacement_args(in_0, in_3):
    # a0=weight(in_0), a1=input(in_3), a2=dummy(in_0 reused), route="rtmpose_gemm"
    return (in_0, in_3, in_0, "rtmpose_gemm")


def replacement_func():
    return dispatch_all