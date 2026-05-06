"""
Pass: conv2d(in_2, in_1, in_0) → stack([x],0) → sum(0) → cat([x, in_3], 1)
Shapes: in_2[N,Cin,H,W], in_1[Cout,Cin,1,1], in_0[Cout],
        in_3[N,Cout,H,W]
"""
import torch
from pass_dir.conv1x1_kernels import fused_dispatch


def pattern(in_0, in_1, in_2, in_3):
    """
    Matches:
        conv2d(in_2, in_1, in_0, ...)   [branch input = in_2]
        torch.stack([x], dim=0)
        .sum(dim=0)
        torch.cat([result, in_3], 1)    [other_input = in_3]
    """
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3  = torch.stack([conv2d], dim=0)
    tmp_4  = tmp_3.sum(dim=0)
    tmp_5  = torch.cat([tmp_4, in_3], 1)
    return (tmp_5,)


def replacement_args(in_0, in_1, in_2, in_3):
    # Route "route_a": branch=in_2, other=in_3
    return (in_0, in_1, in_2, in_3, "route_a")


def replacement_func():
    return fused_dispatch