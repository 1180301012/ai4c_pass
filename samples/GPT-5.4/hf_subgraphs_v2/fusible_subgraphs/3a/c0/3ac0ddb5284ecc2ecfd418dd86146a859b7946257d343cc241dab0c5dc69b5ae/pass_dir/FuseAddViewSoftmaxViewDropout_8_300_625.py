import torch
import triton
import triton.language as tl
from pass_dir.fused_attention_mask_softmax_shared import shared_replacement_func


def pattern(in_0, in_1):
    tmp_0 = in_1 + in_0
    tmp_1 = tmp_0.view(8, 300, 625)
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    tmp_3 = tmp_2.view(1, 8, 300, 625)
    tmp_4 = tmp_3.view(8, 300, 625)
    tmp_5 = torch.nn.functional.dropout(tmp_4, p=0.0, training=False)
    return (tmp_5, tmp_3)


def replacement_args(in_0, in_1):
    return (in_0, in_1, 'b300c625')


def replacement_func():
    return shared_replacement_func()