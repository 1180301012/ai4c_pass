"""
Pass: D=32, Hout=48, Wout=48, a=64, b=96, C=256
Matches: coat_lite_medium_384 graphs with reshape(1,256,48,48) + split([64,96,96])
"""
import torch
from pass_dir.coat_kernel import coat_full_replace


def pattern(in_0, in_1, in_2):
    tmp_0 = in_1 @ in_0
    tmp_1 = in_1[:, :, 1:, :]
    tmp_2 = in_2[:, :, 1:, :]
    tmp_3 = tmp_2.transpose(-1, -2)
    tmp_4 = tmp_3.reshape(1, 256, 48, 48)
    tmp_5 = torch.functional.split(tmp_4, [64, 96, 96], dim=1)
    tmp_6 = tmp_5[0]
    tmp_7 = tmp_5[1]
    tmp_8 = tmp_5[2]
    return (tmp_0, tmp_6, tmp_7, tmp_8, tmp_1)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, "32_48_48_64_96")


def replacement_func():
    return coat_full_replace