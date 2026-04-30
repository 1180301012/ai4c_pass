# Pattern definitions for conv2d + max_pool2d fusion
# These patterns match the computation graphs to be optimized

import torch

def pattern_conv_stride2(in_0, in_1):
    """Match conv2d(stride=2,pad=3,kernel=7) + max_pool2d"""
    tmp = torch.conv2d(in_1, in_0, None, (2, 2), (3, 3), (1, 1), 1)
    tmp2 = torch.nn.functional.max_pool2d(tmp, 3, 2, 1, 1, ceil_mode=False, return_indices=False)
    return tmp2


def pattern_conv_stride1(in_0, in_1):
    """Match conv2d(stride=1,pad=1,kernel=3) + max_pool2d"""
    tmp = torch.conv2d(in_1, in_0, None, (1, 1), (1, 1), (1, 1), 1)
    tmp2 = torch.nn.functional.max_pool2d(tmp, 3, 2, 1, 1, ceil_mode=False, return_indices=False)
    return tmp2