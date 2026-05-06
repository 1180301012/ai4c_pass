"""Pass: B=64, H=2, D=128"""
import torch
from pass_dir.shared_kernels import fused_attn_dispatch


def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_3 = linear.view(64, -1, 2, 64)
    tmp_4 = tmp_3.transpose(1, 2)
    sdpa = torch.nn.functional.scaled_dot_product_attention(in_5, in_4, tmp_4, attn_mask=in_2, dropout_p=0.0, is_causal=False)
    tmp_6 = sdpa.transpose(1, 2)
    tmp_7 = tmp_6.reshape(64, 128, 128)
    return (tmp_7,)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_4, in_5, "B64_H2_D128_S128")


def replacement_func():
    return fused_attn_dispatch