"""
Pass: D=16, Hout=28, Wout=28, a=32, b=48, C=128
Matches: coat_lite_mini graphs with reshape(1,128,28,28) + split([32,48,48])
"""
import torch
from pass_dir.coat_kernel import coat_full_replace


def pattern(in_0, in_1, in_2):
    tmp_0 = in_1 @ in_0
    tmp_1 = in_1[:, :, 1:, :]
    tmp_2 = in_2[:, :, 1:, :]
    tmp_3 = tmp_2.transpose(-1, -2)
    tmp_4 = tmp_3.reshape(1, 128, 28, 28)
    tmp_5 = torch.functional.split(tmp_4, [32, 48, 48], dim=1)
    tmp_6 = tmp_5[0]
    tmp_7 = tmp_5[1]
    tmp_8 = tmp_5[2]
    return (tmp_0, tmp_6, tmp_7, tmp_8, tmp_1)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, "16_28_28_32_48")


def replacement_func():
    return coat_full_replace