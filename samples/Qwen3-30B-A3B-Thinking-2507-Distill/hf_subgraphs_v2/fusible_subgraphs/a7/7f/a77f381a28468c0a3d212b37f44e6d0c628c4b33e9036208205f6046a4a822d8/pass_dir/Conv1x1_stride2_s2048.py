"""
Pass: 1x1 conv with stride=2 (matches conv2d node only, no slice).
Replaces torch.conv2d with Triton GEMM. The slice remains in the FX graph.
"""

import torch
from pass_dir.conv1x1_triton_impl import triton_conv1x1_dispatch


def pattern(in_0, in_1):
    conv2d = torch.conv2d(in_1, in_0, None, (2, 2), (0, 0), (1, 1), 1)
    return conv2d


def replacement_args(in_0, in_1):
    return (in_0, in_1, "stride2")


def replacement_func():
    return triton_conv1x1_dispatch