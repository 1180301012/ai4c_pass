"""
Pass: FuseConv1x1StackSumCat_conv3_cat2

Matches the pattern:
    conv2d = torch.conv2d(in_3, in_1, in_0, (1,1), (0,0), (1,1), 1)
    tmp_3  = torch.stack([conv2d], dim=0)
    tmp_4  = tmp_3.sum(dim=0)
    tmp_5  = torch.cat([tmp_4, in_2], 1)
    return tmp_5

The stack([x], dim=0).sum(dim=0) is a mathematical no-op for a single tensor.
The replacement fuses the whole sequence into a single Triton GEMM + copy kernel.

Applies to graphs:
  float32/5/start387_end391_20
  bfloat16/4/start387_end391_20
"""

import torch
from pass_dir.shared_conv1x1_cat_kernel import fused_conv1x1_cat


def pattern(in_0, in_1, in_2, in_3):
    """
    Matches: conv2d(in_3, weight=in_1, bias=in_0) → stack → sum → cat(in_2)
    in_0 : bias   [C_out]
    in_1 : weight [C_out, C_in, 1, 1]
    in_2 : cat tensor  [N, C_cat, H, W]
    in_3 : conv input  [N, C_in, H, W]
    """
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3  = torch.stack([conv2d], dim=0)
    tmp_4  = tmp_3.sum(dim=0)
    tmp_5  = torch.cat([tmp_4, in_2], 1)
    return tmp_5


def replacement_args(in_0, in_1, in_2, in_3):
    # Map to (bias, weight, conv_in=in_3, cat_in=in_2)
    return (in_0, in_1, in_3, in_2)


def replacement_func():
    return fused_conv1x1_cat