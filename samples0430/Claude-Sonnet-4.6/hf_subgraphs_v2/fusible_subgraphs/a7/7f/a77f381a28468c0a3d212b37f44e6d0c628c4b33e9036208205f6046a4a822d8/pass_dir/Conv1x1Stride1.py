"""
Pass: Replace  torch.conv2d(..., stride=(1,1), padding=(0,0), dilation=(1,1), groups=1)
      with a Triton direct-NCHW 1x1 conv kernel.

Uses the shared dispatch_conv1x1 replacement_func with route="s1".
"""

import torch
from pass_dir.conv1x1_kernels import dispatch_conv1x1


def pattern(in_0, in_1):
    """
    in_0: weight  [Cout, Cin, 1, 1]
    in_1: input   [N,    Cin, H, W]
    Matches: conv2d with stride=(1,1), no bias, no padding, no dilation, groups=1.
    """
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    return conv2d


def replacement_args(in_0, in_1):
    # Append route string so dispatch_conv1x1 selects stride-1 path
    return (in_0, in_1, "s1")


def replacement_func():
    return dispatch_conv1x1