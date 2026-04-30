import torch
import triton
import triton.language as tl

from pass_dir.shared_fused_attn_tail import fused_matmul_vsplit


def pattern(in_0, in_1, in_2):
    matmul = in_1 @ in_0
    tmp_1 = in_1[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))]
    tmp_2 = in_2[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))]
    tmp_3 = tmp_2.transpose(-1, -2)
    side = tmp_2.shape[2]
    channels = tmp_2.shape[1] * tmp_2.shape[3]
    tmp_4 = tmp_3.reshape(1, channels, side, side)
    d = tmp_2.shape[3]
    split = torch.functional.split(tmp_4, [2 * d, 3 * d, 3 * d], dim=1)
    tmp_6 = split[0]
    tmp_7 = split[1]
    tmp_8 = split[2]
    return (matmul, tmp_6, tmp_7, tmp_8, tmp_1)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return fused_matmul_vsplit