"""
Pass: [4,4,512,8] → permute(0,2,1,3)+contiguous → view(4,512,32)
Covers: bfloat16/5, float32/5
"""
import torch
from pass_dir.shared_conv import triton_permute_contiguous_view


def pattern(in_3):
    tmp_3 = in_3.permute(0, 2, 1, 3)
    tmp_4 = tmp_3.contiguous()
    tmp_5 = tmp_4.view(4, 512, 32)
    return (tmp_5,)


def replacement_args(in_3):
    return (in_3,)


def replacement_func():
    return triton_permute_contiguous_view