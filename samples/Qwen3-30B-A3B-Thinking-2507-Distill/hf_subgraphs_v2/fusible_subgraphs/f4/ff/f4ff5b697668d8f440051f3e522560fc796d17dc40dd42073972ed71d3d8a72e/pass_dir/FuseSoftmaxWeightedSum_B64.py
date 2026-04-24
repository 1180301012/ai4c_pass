"""
Fused pass for B=64: weighted sums starting from pre-computed softmax output tmp_3.
Pattern skips softmax, starts from tmp_3=[64,17,64,64].
"""

import torch
import triton
import triton.language as tl
from pass_dir.shared_kernel import sw_dispatch


def pattern(tmp_3, in_0, in_1):
    tmp_4 = tmp_3.mul(in_0)
    tmp_5 = tmp_4.reshape(64, 17, -1)
    tmp_6 = torch.sum(tmp_5, dim=2, keepdim=True)
    tmp_7 = tmp_3.mul(in_1)
    tmp_8 = tmp_7.reshape(64, 17, -1)
    tmp_9 = torch.sum(tmp_8, dim=2, keepdim=True)
    tmp_10 = torch.cat([tmp_6, tmp_9], dim=-1)
    return tmp_10


def replacement_args(tmp_3, in_0, in_1):
    return (tmp_3, in_0, in_1, "B64")


def replacement_func():
    return sw_dispatch