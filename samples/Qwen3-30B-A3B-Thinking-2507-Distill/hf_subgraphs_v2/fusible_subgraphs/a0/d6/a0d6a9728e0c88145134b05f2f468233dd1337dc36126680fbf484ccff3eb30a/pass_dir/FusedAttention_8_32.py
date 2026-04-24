import torch
from pass_dir.attn_kernels_shared import fused_attn_weighted_sum


def pattern(in_0, in_1):
    # in_0 = attn_weights [H, 1, 1], in_1 = value_states [H, 1, D]
    bmm_1 = torch.bmm(in_0, in_1)
    tmp_4 = bmm_1.view(1, 8, 1, 32)
    tmp_5 = tmp_4.transpose(1, 2)
    tmp_6 = tmp_5.reshape(1, 1, 256)
    return tmp_6


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return fused_attn_weighted_sum