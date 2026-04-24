"""
Pass: fused linear+view+transpose+SDPA+transpose+reshape
Matches: float16/8 bert-tiny L-2-H-512, view(128,-1,8,64), reshape(128,64,512)
"""
import torch
import triton
import triton.language as tl
import math
from pass_dir.shared_fused_kernel import fused_linear_attn


def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_3 = linear.view(128, -1, 8, 64)
    tmp_4 = tmp_3.transpose(1, 2)
    scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(
        in_5, in_4, tmp_4, attn_mask=in_2, dropout_p=0.0, is_causal=False
    )
    tmp_6 = scaled_dot_product_attention.transpose(1, 2)
    tmp_7 = tmp_6.reshape(128, 64, 512)
    return tmp_7


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_3, in_1, in_0, in_5, in_4, in_2)


def replacement_func():
    return fused_linear_attn