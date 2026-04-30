import torch
import triton
import triton.language as tl
from pass_dir.flash_attention_common import flash_attention_qk_softmax_v_reshape


def pattern(attn_probs, in_2):
    matmul_1 = torch.matmul(attn_probs, in_2)
    tmp_6 = matmul_1.transpose(1, 2)
    tmp_7 = tmp_6.contiguous()
    tmp_8 = tmp_7.reshape(1, 257, -1)
    tmp_9 = tmp_8.contiguous()
    return tmp_9


def replacement_args(attn_probs, in_2):
    return (attn_probs, in_2, "matmul_tail")


def replacement_func():
    return flash_attention_qk_softmax_v_reshape