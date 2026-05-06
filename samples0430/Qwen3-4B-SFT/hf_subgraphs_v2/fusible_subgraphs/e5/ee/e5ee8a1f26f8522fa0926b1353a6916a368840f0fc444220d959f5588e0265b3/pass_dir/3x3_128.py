"""
Pattern: depthwise 3x3 conv (groups=128, pad=(1,1), stride=(1,1)) + GELU + dropout(p=0)
Matches: hf-tiny-VanForImageClassification start17, float32 & float16/bfloat16
"""
import torch
from .kernels import fused_dw_conv_gelu_dropout as _d


def pattern(bias, weight, x):
    conv = torch.conv2d(x, weight, bias, (1, 1), (1, 1), (1, 1), 128)
    gelu = torch.nn.functional.gelu(conv)
    dp   = torch.nn.functional.dropout(gelu, 0.0, False, False)
    return dp


def replacement_args(bias, weight, x):
    return (bias, weight, x, "3x3_128")


def replacement_func():
    return _d