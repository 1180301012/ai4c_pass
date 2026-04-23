import torch
import triton
import triton.language as tl
from pass_dir.fused_attention_mid_common import fused_attention


def pattern(in_0, in_1, in_2):
    matmul = torch.matmul(in_0, in_1)
    tmp_1 = matmul / 5.656854249492381
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.0, False, False)
    matmul_1 = torch.matmul(tmp_3, in_2)
    return matmul_1


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, 1.0 / 5.656854249492381)


def replacement_func():
    return fused_attention