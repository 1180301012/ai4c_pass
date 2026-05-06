"""
Pass: 1x1 conv (stride=2) + channel slice [:, :2048, :, :] + return (full, slice)
Matches: NCHW 1x1 convolution with stride=2 and 2048 output channels.
Returns full conv out [B, 2304, H/2, W/2] and tmp_2 [B, 2048, H/2, W/2].
"""
import torch
from pass_dir.kernel_lib import conv1x1_slice_dispatch


def pattern(in_0, in_1):
    conv2d = torch.conv2d(in_1, in_0, None, (2, 2), (0, 0), (1, 1), 1)
    tmp_2 = conv2d[(slice(None, None, None), slice(None, 2048, None), slice(None, None, None), slice(None, None, None))]
    return (conv2d, tmp_2)


def replacement_args(in_0, in_1):
    return (in_1, in_0, "s2_2048r1")


def replacement_func():
    return conv1x1_slice_dispatch