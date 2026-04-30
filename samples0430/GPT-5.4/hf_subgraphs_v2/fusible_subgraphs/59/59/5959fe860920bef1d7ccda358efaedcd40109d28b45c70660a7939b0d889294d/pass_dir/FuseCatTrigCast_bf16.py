import torch
import triton
import triton.language as tl
from pass_dir.shared_rmsnorm import shared_dispatch


def pattern(in_1):
    tmp_1 = torch.cat((in_1, in_1), dim=-1)
    tmp_2 = tmp_1.cos()
    tmp_3 = tmp_2 * 1.0
    tmp_4 = tmp_1.sin()
    tmp_5 = tmp_4 * 1.0
    tmp_6 = tmp_3.to(dtype=torch.bfloat16)
    tmp_7 = tmp_5.to(dtype=torch.bfloat16)
    return tmp_6, tmp_7


def replacement_args(in_1):
    return (in_1, "cat_trig_bf16")


def replacement_func():
    return shared_dispatch