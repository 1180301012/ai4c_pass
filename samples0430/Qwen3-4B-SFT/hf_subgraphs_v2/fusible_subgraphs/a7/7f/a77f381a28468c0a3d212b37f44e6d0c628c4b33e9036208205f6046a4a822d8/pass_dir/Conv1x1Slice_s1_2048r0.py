"""
Pass: 1x1 conv (stride=1) + channel slice [:, :2048, :, :] + return (slice, full)
Matches: DPN107 and DPN68 stride-1 blocks with 2048 output channels.
"""
import torch
from pass_dir.kernel_lib import conv1x1_slice_dispatch


def pattern(in_0, in_1):
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_2 = conv2d[(slice(None, None, None), slice(None, 2048, None), slice(None, None, None), slice(None, None, None))]
    return (tmp_2, conv2d)


def replacement_args(in_0, in_1):
    return (in_1, in_0, "s1_2048r0")


def replacement_func():
    return conv1x1_slice_dispatch