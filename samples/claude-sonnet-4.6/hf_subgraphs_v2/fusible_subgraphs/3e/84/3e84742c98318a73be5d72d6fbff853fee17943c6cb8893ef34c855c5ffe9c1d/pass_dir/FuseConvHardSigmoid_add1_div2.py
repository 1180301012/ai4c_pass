"""
Full-fusion pass: conv2d(1x1) + (x+1.0)/2.0 + clamp(0,1) + broadcast-mul.

Replaces the ENTIRE subgraph (including conv2d) with two back-to-back Triton
kernels (GEMM + elementwise), eliminating the CUDA-stream stall that occurs
when torch.compile places a graph break AFTER a compiled conv2d.
"""

import torch
from pass_dir.shared_conv_hardsigmoid_kernels import _dispatch_full


def pattern(in_0, in_1, in_2, in_3):
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d + 1.0
    tmp_4 = tmp_3 / 2.0
    tmp_5 = tmp_4.clamp_(0.0, 1.0)
    tmp_6 = in_2 * tmp_5
    return tmp_6


def replacement_args(in_0, in_1, in_2, in_3):
    # in_0 = bias   [Cout]
    # in_1 = weight [Cout, Cin, 1, 1]
    # in_2 = feature map [B, Cout, H, W]
    # in_3 = SE input    [B, Cin, 1, 1]
    return (in_0, in_1, in_2, in_3, "add1_div2")


def replacement_func():
    return _dispatch_full