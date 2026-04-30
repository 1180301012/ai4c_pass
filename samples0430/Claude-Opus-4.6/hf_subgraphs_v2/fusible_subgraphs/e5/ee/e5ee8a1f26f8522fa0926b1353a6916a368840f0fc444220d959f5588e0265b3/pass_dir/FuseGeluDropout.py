import torch
from pass_dir.shared_kernel import fused_dispatch


def pattern(bias, weight, input, groups):
    conv = torch.conv2d(input, weight, bias, (1, 1), (1, 1), (1, 1), groups)
    gelu_out = torch.nn.functional.gelu(conv)
    drop_out = torch.nn.functional.dropout(gelu_out, 0.0, False, False)
    return drop_out


def replacement_args(bias, weight, input, groups):
    return (input, weight, bias, "dw_conv3x3_gelu")


def replacement_func():
    return fused_dispatch