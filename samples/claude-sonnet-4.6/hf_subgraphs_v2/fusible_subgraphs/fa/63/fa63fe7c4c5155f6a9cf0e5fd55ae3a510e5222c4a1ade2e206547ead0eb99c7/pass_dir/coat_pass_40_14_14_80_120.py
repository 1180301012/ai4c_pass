"""
Pass: D=40, Hout=14, Wout=14, a=80, b=120, C=320
Matches: coat_lite_medium graphs with reshape(1,320,14,14) + split([80,120,120])
"""
import torch
from pass_dir.coat_kernel import coat_full_replace


def pattern(in_0, in_1, in_2):
    tmp_0 = in_1 @ in_0
    tmp_1 = in_1[:, :, 1:, :]
    tmp_2 = in_2[:, :, 1:, :]
    tmp_3 = tmp_2.transpose(-1, -2)
    tmp_4 = tmp_3.reshape(1, 320, 14, 14)
    tmp_5 = torch.functional.split(tmp_4, [80, 120, 120], dim=1)
    tmp_6 = tmp_5[0]
    tmp_7 = tmp_5[1]
    tmp_8 = tmp_5[2]
    return (tmp_0, tmp_6, tmp_7, tmp_8, tmp_1)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, "40_14_14_80_120")


def replacement_func():
    return coat_full_replace