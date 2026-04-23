import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (1, 1), (1, 1), 1)
    tmp_2 = torch.nn.functional.max_pool2d(conv2d, 3, 2, 1, 1, ceil_mode=False, return_indices=False)
    return tmp_2


def replacement_args(in_0, in_1):
    return (in_0, in_1, "conv2d_stride1_pad1_maxpool")


# ---- Import the shared Triton kernel and dispatch function ----
# The kernel and dispatch are defined in FuseConv2DMaxPool2D_stride2_pad3.py
# We share the same replacement_func to satisfy the output_pass_replacement_func_limit constraint

from pass_dir.FuseConv2DMaxPool2D_stride2_pad3 import fused_conv2d_maxpool_dispatch

def replacement_func():
    return fused_conv2d_maxpool_dispatch