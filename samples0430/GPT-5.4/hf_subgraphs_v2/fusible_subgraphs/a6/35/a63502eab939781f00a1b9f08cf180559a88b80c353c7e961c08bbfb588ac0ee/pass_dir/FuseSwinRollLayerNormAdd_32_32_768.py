import os
import sys
import torch
import triton
import triton.language as tl

_PASS_DIR = os.path.dirname(__file__)
if _PASS_DIR not in sys.path:
    sys.path.append(_PASS_DIR)

from shared_swin_roll_ln_add import shared_swin_roll_layernorm_add


def pattern(in_3):
    tmp_2 = in_3.contiguous()
    tmp_3 = tmp_2.view(-1, 32, 32, 768)
    tmp_4 = torch.roll(tmp_3, shifts=(4, 4), dims=(1, 2))
    tmp_5 = tmp_4.view(1, 1024, 768)
    return tmp_5


def replacement_args(in_3):
    return (in_3, "swin_roll_prefix_32_32_768")


def replacement_func():
    return shared_swin_roll_layernorm_add