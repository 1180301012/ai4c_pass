import torch
import triton
import triton.language as tl
from pass_dir.flash_attn_core import flash_attn_forward


def pattern(in_0, in_1, in_2):
    tmp_0 = torch.matmul(in_0, in_1)
    tmp_1 = tmp_0 * 1.0
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1, dtype=torch.float32)
    tmp_3 = tmp_2.to(torch.float32)
    tmp_4 = torch.nn.functional.dropout(tmp_3, p=0.0, training=False)
    tmp_5 = torch.matmul(tmp_4, in_2)
    tmp_6 = tmp_5.transpose(1, 2)
    tmp_7 = tmp_6.contiguous()
    tmp_8 = tmp_7.reshape(1, 257, -1)
    tmp_9 = tmp_8.contiguous()
    return tmp_9


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return flash_attn_forward