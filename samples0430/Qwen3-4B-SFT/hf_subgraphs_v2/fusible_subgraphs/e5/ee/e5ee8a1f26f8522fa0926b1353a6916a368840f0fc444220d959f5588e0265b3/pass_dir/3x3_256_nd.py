"""3x3 dw conv groups=256 + GELU + dropout p=0  (nd variant)"""
import torch
from .kernels import fused_dw_conv_gelu_dropout as _d

def pattern(bias, weight, x):
    conv = torch.conv2d(x, weight, bias, (1, 1), (1, 1), (1, 1), 256)
    gelu = torch.nn.functional.gelu(conv)
    dp   = torch.nn.functional.dropout(gelu, 0.0, False, False)
    return dp

def replacement_args(bias, weight, x):
    return (bias, weight, x, "3x3_256")

def replacement_func():
    return _d