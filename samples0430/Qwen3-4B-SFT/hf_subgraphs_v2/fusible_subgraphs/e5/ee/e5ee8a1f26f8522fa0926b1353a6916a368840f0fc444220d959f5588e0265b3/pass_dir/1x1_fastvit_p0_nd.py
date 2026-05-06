"""1x1 conv groups=1 + GELU exact (no dropout), fastvit-style"""
import torch
from .kernels import fused_dw_conv_gelu_dropout as _d

def pattern(bias, weight, x):
    conv = torch.conv2d(x, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    gelu = torch.nn.functional.gelu(conv, approximate = 'none')
    return gelu

def replacement_args(bias, weight, x):
    return (bias, weight, x, "1x1_1")

def replacement_func():
    return _d