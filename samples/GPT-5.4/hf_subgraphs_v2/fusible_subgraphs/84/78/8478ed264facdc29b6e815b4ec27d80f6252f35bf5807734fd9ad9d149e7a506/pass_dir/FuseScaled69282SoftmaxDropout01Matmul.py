import torch
import triton
import triton.language as tl
from pass_dir.fused_attention_mid_common import fused_attention


def pattern(in_0, in_1, in_2):
    tmp_0 = torch.matmul(in_0, in_1)
    tmp_1 = tmp_0 / 6.928203230275509
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.1, False, False)
    tmp_4 = torch.matmul(tmp_3, in_2)
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, 1.0 / 6.928203230275509)


def replacement_func():
    return fused_attention