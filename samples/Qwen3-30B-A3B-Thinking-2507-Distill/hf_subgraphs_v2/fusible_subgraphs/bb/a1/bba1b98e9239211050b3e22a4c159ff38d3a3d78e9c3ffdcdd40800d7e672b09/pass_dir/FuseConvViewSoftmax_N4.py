"""
Fuse: conv2d(1×1) + view(N, 1, 4096) + softmax  →  single Triton kernel
Matches the bfloat16/2 graph where batch N=4.
The view(4, 1, -1) resolves to view(4, 1, 4096) at FX-trace time.
"""
import torch
from pass_dir.conv1x1_softmax_kernel import fused_conv1x1_softmax


def pattern(bias, weight, inp):
    conv    = torch.conv2d(inp, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    viewed  = conv.view(4, 1, -1)
    out     = viewed.softmax(dim=-1)
    return out


def replacement_args(bias, weight, inp):
    return (bias, weight, inp)


def replacement_func():
    return fused_conv1x1_softmax