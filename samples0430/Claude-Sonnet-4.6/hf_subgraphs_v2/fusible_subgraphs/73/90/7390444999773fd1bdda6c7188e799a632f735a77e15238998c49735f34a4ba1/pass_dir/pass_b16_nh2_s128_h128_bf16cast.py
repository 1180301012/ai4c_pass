import torch
from pass_dir.shared_kernels import _dispatch

def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_3 = linear.view(16, -1, 2, 64)
    tmp_4 = tmp_3.transpose(1, 2)
    to = tmp_4.to(torch.bfloat16)
    scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(in_5, in_4, to, attn_mask=in_2, dropout_p=0.0, is_causal=False)
    tmp_6 = scaled_dot_product_attention.transpose(1, 2)
    tmp_7 = tmp_6.reshape(16, 128, 128)
    return (tmp_7,)

def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5, "b16_nh2_s128_h128_bf16cast")

def replacement_func():
    return _dispatch