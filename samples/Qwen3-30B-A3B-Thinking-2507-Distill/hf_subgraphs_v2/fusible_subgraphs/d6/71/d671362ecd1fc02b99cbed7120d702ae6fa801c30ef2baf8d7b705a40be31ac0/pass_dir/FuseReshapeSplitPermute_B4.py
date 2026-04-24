import torch
import triton
from pass_dir.reshape_split_permute_kernel import qkv_fused_wrapper


def pattern(linear_out, in_0):
    tmp_4  = linear_out.reshape(4, 49, 8, -1)
    split  = tmp_4.split([32, 32, 128], dim=3)
    tmp_6  = split[0]
    tmp_7  = split[1]
    tmp_8  = split[2]
    tmp_9  = tmp_6.permute(0, 2, 1, 3)
    tmp_10 = tmp_7.permute(0, 2, 1, 3)
    tmp_11 = tmp_8.permute(0, 2, 1, 3)
    tmp_12 = in_0.to(device(type='cuda', index=0))
    tmp_13 = tmp_10.transpose(-2, -1)
    return (tmp_9, tmp_12, tmp_13, tmp_11)


def replacement_args(linear_out, in_0):
    return (linear_out, in_0)


def replacement_func():
    return qkv_fused_wrapper