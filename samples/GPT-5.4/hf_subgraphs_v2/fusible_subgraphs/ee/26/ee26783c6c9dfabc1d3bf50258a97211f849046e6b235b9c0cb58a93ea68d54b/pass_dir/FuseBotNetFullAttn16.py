import torch
import triton
import triton.language as tl

from pass_dir.botnet_hybrid_dispatch import botnet_hybrid_dispatch


def pattern(in_0, in_1, in_2, in_3, in_4):
    matmul = in_1 @ in_3
    tmp_1 = matmul.reshape(-1, 16, 31)
    tmp_2 = torch._C._nn.pad(tmp_1, [0, 1], 'constant', None)
    tmp_3 = tmp_2.flatten(1)
    tmp_4 = torch._C._nn.pad(tmp_3, [0, 15], 'constant', None)
    tmp_5 = tmp_4.reshape(-1, 17, 31)
    tmp_6 = tmp_5[(slice(None, None, None), slice(None, 16, None), slice(15, None, None))]
    tmp_7 = tmp_6.reshape(4, 16, 1, 16, 16)
    tmp_8 = tmp_7.expand(-1, -1, 16, -1, -1)
    tmp_9 = tmp_8.permute((0, 3, 1, 4, 2))
    tmp_10 = tmp_9 + in_2
    tmp_11 = tmp_10.reshape(4, 256, 256)
    tmp_12 = in_0 + tmp_11
    tmp_13 = tmp_12.softmax(dim=-1)
    matmul_1 = tmp_13 @ in_4
    tmp_15 = matmul_1.transpose(-1, -2)
    return tmp_15


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4, "botnet_full_attn_16")


def replacement_func():
    return botnet_hybrid_dispatch