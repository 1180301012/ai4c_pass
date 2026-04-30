import torch
from torch import device
from pass_dir.shared_qkv_impl import qkv_linear_split_permute


def pattern(in_0, in_1, in_2, in_3):
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = in_2
    tmp_3 = torch.nn.functional.linear(in_3, tmp_2, tmp_1)
    tmp_4 = tmp_3.reshape(512, 49, 8, -1)
    tmp_5 = tmp_4.split([32, 32, 128], dim=3)
    tmp_6 = tmp_5[0]
    tmp_7 = tmp_5[1]
    tmp_8 = tmp_5[2]
    tmp_9 = tmp_6.permute(0, 2, 1, 3)
    tmp_10 = tmp_7.permute(0, 2, 1, 3)
    tmp_11 = tmp_8.permute(0, 2, 1, 3)
    tmp_12 = tmp_0.to(device(type='cuda', index=0))
    tmp_13 = tmp_10.transpose(-2, -1)
    return (tmp_9, tmp_12, tmp_13, tmp_11)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3, 'b512')


def replacement_func():
    return qkv_linear_split_permute