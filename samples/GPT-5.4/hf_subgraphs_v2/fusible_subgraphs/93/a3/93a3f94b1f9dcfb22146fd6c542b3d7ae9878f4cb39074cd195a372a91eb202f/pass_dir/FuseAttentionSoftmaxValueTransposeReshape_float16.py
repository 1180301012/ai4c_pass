import torch
import triton
import triton.language as tl
from pass_dir.shared_fused_attention import fused_attention_dispatch


def pattern(in_0: torch.Tensor, in_1, in_2):
    matmul = torch.matmul(in_0, in_1)
    tmp_1 = matmul * 1.0
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1, dtype=torch.float32)
    tmp_4 = torch.nn.functional.dropout(tmp_2, p=0.0, training=False)
    to = tmp_4.to(torch.float16)
    matmul_1 = torch.matmul(to, in_2)
    tmp_6 = matmul_1.transpose(1, 2)
    tmp_7 = tmp_6.contiguous()
    tmp_8 = tmp_7.reshape(1, 257, -1)
    tmp_9 = tmp_8.contiguous()
    return tmp_9


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, 'fp16')


def replacement_func():
    return fused_attention_dispatch