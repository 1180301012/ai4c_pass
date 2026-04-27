"""
Pass: D=27, Hout=28, Wout=28, a=54, b=81, C=216
Matches: coat_mini graphs with reshape(1,216,28,28) + split([54,81,81])
"""
import torch
from pass_dir.coat_kernel import coat_full_replace


def pattern(in_0, in_1, in_2):
    tmp_0 = in_1 @ in_0
    tmp_1 = in_1[:, :, 1:, :]
    tmp_2 = in_2[:, :, 1:, :]
    tmp_3 = tmp_2.transpose(-1, -2)
    tmp_4 = tmp_3.reshape(1, 216, 28, 28)
    tmp_5 = torch.functional.split(tmp_4, [54, 81, 81], dim=1)
    tmp_6 = tmp_5[0]
    tmp_7 = tmp_5[1]
    tmp_8 = tmp_5[2]
    return (tmp_0, tmp_6, tmp_7, tmp_8, tmp_1)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, "27_28_28_54_81")


def replacement_func():
    return coat_full_replace