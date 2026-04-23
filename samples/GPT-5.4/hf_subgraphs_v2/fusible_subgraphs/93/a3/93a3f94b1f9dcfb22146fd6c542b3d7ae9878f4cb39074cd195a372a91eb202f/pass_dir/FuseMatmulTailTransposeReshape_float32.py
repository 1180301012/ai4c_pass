import torch
import triton
import triton.language as tl
from pass_dir.shared_tail_matmul import fused_matmul_tail


def pattern(attn, in_2):
    tmp_5 = torch.matmul(attn, in_2)
    tmp_6 = tmp_5.transpose(1, 2)
    tmp_7 = tmp_6.contiguous()
    tmp_8 = tmp_7.reshape(1, 257, -1)
    tmp_9 = tmp_8.contiguous()
    return tmp_9


def replacement_args(attn, in_2):
    return (attn, in_2)


def replacement_func():
    return fused_matmul_tail