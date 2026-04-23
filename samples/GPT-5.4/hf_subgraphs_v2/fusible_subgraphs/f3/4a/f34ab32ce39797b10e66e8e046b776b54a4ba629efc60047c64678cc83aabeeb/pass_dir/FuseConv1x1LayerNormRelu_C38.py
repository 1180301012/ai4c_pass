import torch
import triton
import triton.language as tl
from pass_dir.fused_conv1x1_layernorm_relu_shared import fused_conv1x1_layernorm_relu


def pattern(in_0, in_1, in_2, in_3, in_4):
    conv2d = torch.conv2d(in_4, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_5 = torch.nn.functional.layer_norm(conv2d, (38, 1, 1), in_3, in_2, 1e-05)
    tmp_6 = torch.nn.functional.relu(tmp_5, inplace=True)
    return tmp_6


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


def replacement_func():
    return fused_conv1x1_layernorm_relu