"""
Pass: FusedTinyAttn_16_64_64
Fuses: bmm -> softmax -> dropout(p=0) -> bmm -> view(1,16,1,64) -> transpose(1,2) -> reshape(1,1,1024)
Target shapes: Q=[16,64,1], K=[16,64,1], V=[16,1,64]  -> output [1,1,1024]
"""
import torch
from pass_dir.triton_tiny_attn_shared import fused_tiny_attn_b16_h64_d64


def pattern(in_0, in_1, in_2):
    bmm   = torch.bmm(in_0, in_1)
    tmp_1 = torch.nn.functional.softmax(bmm, dim=-1)
    tmp_2 = torch.nn.functional.dropout(tmp_1, p=0.0, training=False)
    bmm_1 = torch.bmm(tmp_2, in_2)
    tmp_4 = bmm_1.view(1, 16, 1, 64)
    tmp_5 = tmp_4.transpose(1, 2)
    tmp_6 = tmp_5.reshape(1, 1, 1024)
    return (tmp_6,)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return fused_tiny_attn_b16_h64_d64