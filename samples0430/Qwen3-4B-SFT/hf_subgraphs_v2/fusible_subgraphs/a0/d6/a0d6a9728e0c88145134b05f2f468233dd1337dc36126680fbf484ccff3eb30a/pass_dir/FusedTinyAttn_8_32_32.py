"""
Pass: FusedTinyAttn_8_32_32
Fuses: bmm -> softmax -> dropout(p=0) -> bmm -> view(1,8,1,32) -> transpose(1,2) -> reshape(1,1,256)
Target shapes: Q=[8,32,1], K=[8,32,1], V=[8,1,32]  -> output [1,1,256]
"""
import torch
from pass_dir.triton_tiny_attn_shared import fused_tiny_attn_b8_h32_d32


def pattern(in_0, in_1, in_2):
    bmm   = torch.bmm(in_0, in_1)
    tmp_1 = torch.nn.functional.softmax(bmm, dim=-1)
    tmp_2 = torch.nn.functional.dropout(tmp_1, p=0.0, training=False)
    bmm_1 = torch.bmm(tmp_2, in_2)
    tmp_4 = bmm_1.view(1, 8, 1, 32)
    tmp_5 = tmp_4.transpose(1, 2)
    tmp_6 = tmp_5.reshape(1, 1, 256)
    return (tmp_6,)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return fused_tiny_attn_b8_h32_d32