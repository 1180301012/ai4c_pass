import torch
import triton
import triton.language as tl


# Import the shared wrapper from the other pass file
from pass_dir.FusedSDPAWithValueProjection import fused_sdpa_value_wrapper


def pattern_16_128_128_2_64_fp16_to(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Match pattern with .to(float16) cast: linear -> view(16,-1,2,64) -> transpose -> .to(float16) -> sdpa -> transpose -> reshape(16,128,128)
    """
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_3 = linear.view(16, -1, 2, 64)
    tmp_4 = tmp_3.transpose(1, 2)
    to = tmp_4.to(torch.float16)
    scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(in_5, in_4, to, attn_mask=in_2, dropout_p=0.0, is_causal=False)
    tmp_6 = scaled_dot_product_attention.transpose(1, 2)
    tmp_7 = tmp_6.reshape(16, 128, 128)
    return tmp_7


def replacement_args_16_128_128_2_64_fp16_to(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5, 0.0, False, "16_128_128_2_64_fp16_to")


def pattern_1_12_256_4_64_fp16_to(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Match pattern with .to(float16) cast: linear -> view(1,-1,4,64) -> transpose -> .to(float16) -> sdpa -> transpose -> reshape(1,12,256)
    """
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_3 = linear.view(1, -1, 4, 64)
    tmp_4 = tmp_3.transpose(1, 2)
    to = tmp_4.to(torch.float16)
    scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(in_5, in_4, to, attn_mask=in_2, dropout_p=0.0, is_causal=False)
    tmp_6 = scaled_dot_product_attention.transpose(1, 2)
    tmp_7 = tmp_6.reshape(1, 12, 256)
    return tmp_7


def replacement_args_1_12_256_4_64_fp16_to(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5, 0.0, False, "1_12_256_4_64_fp16_to")


def pattern_128_64_512_8_64_fp16_to(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Match pattern with .to(float16) cast: linear -> view(128,-1,8,64) -> transpose -> .to(float16) -> sdpa -> transpose -> reshape(128,64,512)
    """
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_3 = linear.view(128, -1, 8, 64)
    tmp_4 = tmp_3.transpose(1, 2)
    to = tmp_4.to(torch.float16)
    scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(in_5, in_4, to, attn_mask=in_2, dropout_p=0.0, is_causal=False)
    tmp_6 = scaled_dot_product_attention.transpose(1, 2)
    tmp_7 = tmp_6.reshape(128, 64, 512)
    return tmp_7


def replacement_args_128_64_512_8_64_fp16_to(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5, 0.0, False, "128_64_512_8_64_fp16_to")


def pattern_1_12_128_2_64_fp16_to(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Match pattern with .to(float16) cast: linear -> view(1,-1,2,64) -> transpose -> .to(float16) -> sdpa -> transpose -> reshape(1,12,128)
    """
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_3 = linear.view(1, -1, 2, 64)
    tmp_4 = tmp_3.transpose(1, 2)
    to = tmp_4.to(torch.float16)
    scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(in_5, in_4, to, attn_mask=in_2, dropout_p=0.0, is_causal=False)
    tmp_6 = scaled_dot_product_attention.transpose(1, 2)
    tmp_7 = tmp_6.reshape(1, 12, 128)
    return tmp_7


def replacement_args_1_12_128_2_64_fp16_to(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5, 0.0, False, "1_12_128_2_64_fp16_to")


def pattern_16_128_128_2_64_bf16_to(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Match pattern with .to(bfloat16) cast: linear -> view(16,-1,2,64) -> transpose -> .to(bfloat16) -> sdpa -> transpose -> reshape(16,128,128)
    """
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_3 = linear.view(16, -1, 2, 64)
    tmp_4 = tmp_3.transpose(1, 2)
    to = tmp_4.to(torch.bfloat16)
    scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(in_5, in_4, to, attn_mask=in_2, dropout_p=0.0, is_causal=False)
    tmp_6 = scaled_dot_product_attention.transpose(1, 2)
    tmp_7 = tmp_6.reshape(16, 128, 128)
    return tmp_7


def replacement_args_16_128_128_2_64_bf16_to(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5, 0.0, False, "16_128_128_2_64_bf16_to")


def pattern_1_12_256_4_64_bf16_to(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Match pattern with .to(bfloat16) cast: linear -> view(1,-1,4,64) -> transpose -> .to(bfloat16) -> sdpa -> transpose -> reshape(1,12,256)
    """
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_3 = linear.view(1, -1, 4, 64)
    tmp_4 = tmp_3.transpose(1, 2)
    to = tmp_4.to(torch.bfloat16)
    scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(in_5, in_4, to, attn_mask=in_2, dropout_p=0.0, is_causal=False)
    tmp_6 = scaled_dot_product_attention.transpose(1, 2)
    tmp_7 = tmp_6.reshape(1, 12, 256)
    return tmp_7


def replacement_args_1_12_256_4_64_bf16_to(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5, 0.0, False, "1_12_256_4_64_bf16_to")


# All patterns share the same replacement function (imported from FusedSDPAWithValueProjection)
def replacement_func():
    return fused_sdpa_value_wrapper