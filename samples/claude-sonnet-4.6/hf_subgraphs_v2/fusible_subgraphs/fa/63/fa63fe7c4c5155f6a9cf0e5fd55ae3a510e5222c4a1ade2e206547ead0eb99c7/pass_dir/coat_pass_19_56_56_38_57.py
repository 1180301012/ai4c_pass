"""
Pass: D=19, Hout=56, Wout=56, a=38, b=57, C=152
Matches: coat_mini graphs with reshape(1,152,56,56) + split([38,57,57])
"""
import torch
from pass_dir.coat_kernel import coat_full_replace


def pattern(in_0, in_1, in_2):
    tmp_0 = in_1 @ in_0
    tmp_1 = in_1[:, :, 1:, :]
    tmp_2 = in_2[:, :, 1:, :]
    tmp_3 = tmp_2.transpose(-1, -2)
    tmp_4 = tmp_3.reshape(1, 152, 56, 56)
    tmp_5 = torch.functional.split(tmp_4, [38, 57, 57], dim=1)
    tmp_6 = tmp_5[0]
    tmp_7 = tmp_5[1]
    tmp_8 = tmp_5[2]
    return (tmp_0, tmp_6, tmp_7, tmp_8, tmp_1)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, "19_56_56_38_57")


def replacement_func():
    return coat_full_replace