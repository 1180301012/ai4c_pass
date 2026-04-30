import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _kernels import fused_dispatch


def pattern(bias, weight, x):
    conv = torch.conv2d(x, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    reshaped = conv.view(1, 1, -1)
    out = reshaped.softmax(dim=-1)
    return out


def replacement_args(bias, weight, x):
    return (x, weight, bias, "route_b1")


def replacement_func():
    return fused_dispatch