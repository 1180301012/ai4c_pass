import torch
from pass_dir.shared_conv_gelu import fused_conv_gelu_dispatch


def pattern(in_0, in_1, in_2):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (1, 1), (1, 1), 128)
    tmp_3 = torch.nn.functional.gelu(conv2d)
    tmp_4 = torch.nn.functional.dropout(tmp_3, 0.0, False, False)
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, 'depthwise3x3')


def replacement_func():
    return fused_conv_gelu_dispatch