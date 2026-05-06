"""
Pass: FuseConvMaxPool_stride1_pad1
Matches: conv2d(in_1, in_0, None, (1,1), (1,1), (1,1), 1) + max_pool2d(...) followed
by max_pool2d(center_window, 3, 2, 1, 1, ceil_mode=False, return_indices=False)
from resnetv2_18d stems (3x3 conv, stride=1, pad=1).
"""

import torch
from pass_dir.shared_conv_maxpool import fused_conv_maxpool


def pattern(in_0, in_1):
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (1, 1), (1, 1), 1)
    tmp_2 = torch.nn.functional.max_pool2d(conv2d, 3, 2, 1, 1, ceil_mode=False, return_indices=False)
    return (tmp_2,)


def replacement_args(in_0, in_1):
    return (in_1, in_0, "B")


def replacement_func():
    return fused_conv_maxpool