import torch
from pass_dir.conv_gelu_kernels import unified_conv_gelu_dispatch


def pattern(bias, weight, x):
    conv = torch.conv2d(x, weight, bias, (1, 1), (1, 1), (1, 1), 512)
    gelu = torch.nn.functional.gelu(conv)
    out  = torch.nn.functional.dropout(gelu, 0.0, False, False)
    return out


def replacement_args(bias, weight, x):
    return (bias, weight, x, "dw")


def replacement_func():
    return unified_conv_gelu_dispatch