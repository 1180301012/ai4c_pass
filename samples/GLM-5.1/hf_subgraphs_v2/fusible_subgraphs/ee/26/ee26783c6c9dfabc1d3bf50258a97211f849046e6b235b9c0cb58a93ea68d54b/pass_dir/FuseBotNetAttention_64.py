import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4):
    matmul = in_1 @ in_3
    tmp_1 = matmul.reshape(-1, 8, 15)
    tmp_2 = torch.nn.functional.pad(tmp_1, [0, 1], 'constant', None)
    tmp_3 = tmp_2.flatten(1)
    tmp_4 = torch.nn.functional.pad(tmp_3, [0, 7], 'constant', None)
    tmp_5 = tmp_4.reshape(-1, 9, 15)
    tmp_6 = tmp_5[(slice(None, None, None), slice(None, 8, None), slice(7, None, None))]
    tmp_7 = tmp_6.reshape(4, 8, 1, 8, 8)
    tmp_8 = tmp_7.expand(-1, -1, 8, -1, -1)
    tmp_9 = tmp_8.permute((0, 3, 1, 4, 2))
    tmp_10 = tmp_9 + in_2
    tmp_11 = tmp_10.reshape(4, 64, 64)
    tmp_12 = in_0 + tmp_11
    tmp_13 = tmp_12.softmax(dim=-1)
    matmul_1 = tmp_13 @ in_4
    tmp_15 = matmul_1.transpose(-1, -2)
    return (tmp_15,)


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4, "route_64")


from pass_dir.FuseBotNetAttention_256_clean import fused_botnet_attention_dispatch


def replacement_func():
    return fused_botnet_attention_dispatch