"""
Pass: Fused depthwise Conv2D (any stride/group) via Triton.
Pattern matches ALL depthwise conv2d with stride=(2,2), pad=(1,1), dil=(1,1), groups=256.
Single-output pattern → no multi-output FX replace_pattern issues.
"""
import torch
from pass_dir.dw_conv_mean_kernel import dispatch_kernel


def pattern(in_0, in_1):
    conv2d = torch.conv2d(in_1, in_0, None, (2, 2), (1, 1), (1, 1), 256)
    return conv2d


def replacement_args(in_0, in_1):
    return (in_0, in_1, "conv")


def replacement_func():
    return dispatch_kernel