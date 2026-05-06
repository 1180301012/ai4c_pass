"""
Pass: FuseConvMaxPool_stride2_pad3
Matches: conv2d(in_1, in_0, None, (2,2), (3,3), (1,1), 1) + max_pool2d(...) followed
by max_pool2d(center_window, 3, 2, 1, 1, ceil_mode=False, return_indices=False)
from resnetv2_101 stems (7x7 conv, stride=2, pad=3).
"""

import torch
from pass_dir.shared_conv_maxpool import fused_conv_maxpool


def pattern(in_0, in_1):
    conv2d = torch.conv2d(in_1, in_0, None, (2, 2), (3, 3), (1, 1), 1)
    tmp_3 = torch.nn.functional.max_pool2d(conv2d, 3, 2, 1, 1, ceil_mode=False, return_indices=False)
    return (tmp_3,)


def replacement_args(in_0, in_1):
    return (in_1, in_0, "A")


def replacement_func():
    return fused_conv_maxpool