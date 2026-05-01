"""
Pass: Replace  torch.conv2d(..., stride=(2,2), padding=(0,0), dilation=(1,1), groups=1)
      with a Triton direct-NCHW 1x1 conv kernel (stride-2 variant).

Uses the shared dispatch_conv1x1 replacement_func with route="s2".
"""

import torch
from pass_dir.conv1x1_kernels import dispatch_conv1x1


def pattern(in_0, in_1):
    """
    in_0: weight  [Cout, Cin, 1, 1]
    in_1: input   [N,    Cin, H, W]
    Matches: conv2d with stride=(2,2), no bias, no padding, no dilation, groups=1.
    """
    conv2d = torch.conv2d(in_1, in_0, None, (2, 2), (0, 0), (1, 1), 1)
    return conv2d


def replacement_args(in_0, in_1):
    # Append route string so dispatch_conv1x1 selects stride-2 path
    return (in_0, in_1, "s2")


def replacement_func():
    return dispatch_conv1x1