"""
Pass: 1x1 conv (stride=1) + channel slice [:, :512, :, :] + return (slice, full)
"""
import torch
from pass_dir.kernel_lib import conv1x1_slice_dispatch


def pattern(in_0, in_1):
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_2 = conv2d[(slice(None, None, None), slice(None, 512, None), slice(None, None, None), slice(None, None, None))]
    return (tmp_2, conv2d)


def replacement_args(in_0, in_1):
    return (in_1, in_0, "s1_512r0")


def replacement_func():
    return conv1x1_slice_dispatch