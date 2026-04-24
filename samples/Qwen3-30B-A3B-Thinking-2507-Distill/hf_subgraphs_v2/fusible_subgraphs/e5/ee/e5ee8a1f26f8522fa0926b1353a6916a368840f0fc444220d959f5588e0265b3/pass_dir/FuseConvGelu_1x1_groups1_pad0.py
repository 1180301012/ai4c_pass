"""
Fused 1x1 conv + GELU + no-op dropout for groups=1, padding=(0,0).
Matches fastvit: torch.conv2d(in_2, in_1, in_0, (1,1), (0,0), (1,1), 1) -> gelu -> dropout(0.0)
weight shape: [256, 64, 1, 1], input shape: [N, 64, H, W]
"""
import torch
import triton
import triton.language as tl
from pass_dir.conv_gelu_kernel import fused_conv_gelu_dispatch


def pattern(in_0, in_1, in_2):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.nn.functional.gelu(conv2d)
    tmp_4 = torch.nn.functional.dropout(tmp_3, 0.0, False, False)
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, "groups1_pad0")


def replacement_func():
    return fused_conv_gelu_dispatch