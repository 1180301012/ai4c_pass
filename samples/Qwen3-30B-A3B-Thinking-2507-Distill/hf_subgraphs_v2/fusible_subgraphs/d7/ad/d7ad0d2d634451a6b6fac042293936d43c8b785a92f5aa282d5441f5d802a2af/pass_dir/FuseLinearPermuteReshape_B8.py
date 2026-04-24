"""
Pass: fuse linear + permute(0,2,1) + reshape(8, -1, 16, 16)
Matches float16/3 graphs (B=8).
Uses shared dispatch_fused_linear so all passes share the same replacement_func.
"""
import torch
from pass_dir.fused_linear_kernel import dispatch_fused_linear


def pattern(in_0, in_1, in_2):
    """
    Matches:
        linear = torch.nn.functional.linear(in_2, in_1, in_0)
        tmp_3  = linear.permute(0, 2, 1)
        tmp_4  = tmp_3.reshape(8, -1, 16, 16)
    """
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3  = linear.permute(0, 2, 1)
    tmp_4  = tmp_3.reshape(8, -1, 16, 16)
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    # Route "B8" – dispatch wrapper ignores route and uses in_2.shape instead
    return (in_0, in_1, in_2, "route_B8")


def replacement_func():
    return dispatch_fused_linear