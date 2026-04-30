import torch
import triton
import triton.language as tl

from pass_dir.shared_sdpa import sdpa_wrapper


def pattern(in_0, in_1, in_2):
    matmul = torch.matmul(in_0, in_1)
    tmp_1 = matmul / 8.0
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.0, False, False)
    matmul_1 = torch.matmul(tmp_3, in_2)
    tmp_5 = matmul_1.permute(0, 2, 1, 3)
    tmp_6 = tmp_5.contiguous()
    return tmp_6


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, 8.0, 0.0)


def replacement_func():
    return sdpa_wrapper