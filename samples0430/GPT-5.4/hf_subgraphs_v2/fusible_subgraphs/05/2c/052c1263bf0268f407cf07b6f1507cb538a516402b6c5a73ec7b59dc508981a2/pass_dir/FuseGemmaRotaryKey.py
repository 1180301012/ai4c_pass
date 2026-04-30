import torch
import triton
import triton.language as tl
from pass_dir.shared_gemma_rotary import shared_dispatch


def pattern(in_1, in_2, in_4):
    tmp_0 = in_2 * in_1
    tmp_1 = in_2[Ellipsis, slice(None, 128, None)]
    tmp_2 = in_2[Ellipsis, slice(128, None, None)]
    tmp_3 = -tmp_2
    tmp_4 = torch.cat((tmp_3, tmp_1), dim=-1)
    tmp_5 = tmp_4 * in_4
    tmp_6 = tmp_0 + tmp_5
    return tmp_6


def replacement_args(in_1, in_2, in_4):
    return (in_1, in_2, in_4, "rotary_key")


def replacement_func():
    return shared_dispatch