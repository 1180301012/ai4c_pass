import torch
import triton
import triton.language as tl
from pass_dir.fused_depthwise_conv_add_permute_common import fused_dispatch


def pattern(x):
    tmp_3 = x.permute(0, 2, 1, 3)
    tmp_4 = tmp_3.contiguous()
    return tmp_4


def replacement_args(x):
    return (x, "permute_contiguous")


def replacement_func():
    return fused_dispatch