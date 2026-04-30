"""
Pass: Fused depthwise Conv2D (stride=2, groups=384) + spatial mean.
Matches graphs with:
  conv2d = torch.conv2d(in_1, in_0, None, (2,2), (1,1), (1,1), 384)
  tmp_2  = conv2d.mean((2, 3), keepdim=True)
"""

import torch
from pass_dir.dw_conv_mean_kernel import dispatch_kernel


def pattern(in_0, in_1):
    conv2d = torch.conv2d(in_1, in_0, None, (2, 2), (1, 1), (1, 1), 384)
    tmp_2 = conv2d.mean((2, 3), keepdim=True)
    return (conv2d, tmp_2)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return dispatch_kernel