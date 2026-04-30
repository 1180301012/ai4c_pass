import torch
import triton
import triton.language as tl

# Import the shared kernel and dispatch from groups4 pass
from pass_dir.FuseConv2dAddPermute_groups4 import (
    fused_conv2d_add_permute_kernel,
    _fused_kernel_impl,
    fused_conv2d_add_permute_dispatch,
)


def pattern(in_0, in_1, in_2):
    conv2d = torch.conv2d(in_2, in_0, None, (1, 1), (32, 0), (1, 1), 12)
    in_1 += conv2d
    permuted = in_1.permute(0, 2, 1, 3)
    contiguous_result = permuted.contiguous()
    return (contiguous_result,)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, "groups12")


def replacement_func():
    return fused_conv2d_add_permute_dispatch