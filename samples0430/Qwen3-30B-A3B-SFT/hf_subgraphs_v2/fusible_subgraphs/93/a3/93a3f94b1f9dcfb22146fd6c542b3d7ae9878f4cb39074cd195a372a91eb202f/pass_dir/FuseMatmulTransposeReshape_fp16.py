"""
Pass: Fuse second matmul + transpose + reshape  (float16 variant)
Same generic pattern as bf16; attn placeholder matches float16 attention weights.
"""

import torch
from pass_dir.matmul_transpose_impl import matmul_transpose_reshape


def pattern(attn, in_2):
    matmul_1 = torch.matmul(attn, in_2)
    tmp_6    = matmul_1.transpose(1, 2)
    tmp_7    = tmp_6.contiguous()
    tmp_8    = tmp_7.reshape(1, 257, -1)
    tmp_9    = tmp_8.contiguous()
    return (tmp_9,)


def replacement_args(attn, in_2):
    return (attn, in_2)


def replacement_func():
    return matmul_transpose_reshape