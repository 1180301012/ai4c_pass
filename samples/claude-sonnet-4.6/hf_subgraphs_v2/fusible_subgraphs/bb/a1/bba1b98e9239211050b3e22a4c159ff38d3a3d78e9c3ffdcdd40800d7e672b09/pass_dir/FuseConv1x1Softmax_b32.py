"""
Pass: Fuse conv2d (1x1) + view(32,1,-1) + softmax into a single Triton kernel.
Matches graphs with batch size B=32.
"""

import torch
from pass_dir.conv1x1_softmax_kernel import fused_conv1x1_softmax


def pattern(in_0, in_1, in_2):
    conv = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    view = conv.view(32, 1, -1)
    out  = view.softmax(dim=-1)
    return out


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return fused_conv1x1_softmax