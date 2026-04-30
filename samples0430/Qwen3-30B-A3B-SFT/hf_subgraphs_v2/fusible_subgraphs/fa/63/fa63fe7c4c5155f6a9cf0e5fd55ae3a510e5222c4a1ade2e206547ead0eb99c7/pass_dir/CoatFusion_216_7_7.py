import torch
from pass_dir.shared_coat_ops import dispatch_coat_fusion


def pattern(in_2):
    tmp_2 = in_2[slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None)]
    tmp_3 = tmp_2.transpose(-1, -2)
    tmp_4 = tmp_3.reshape(1, 216, 7, 7)
    split = torch.functional.split(tmp_4, [54, 81, 81], dim=1)
    tmp_6 = split[0]
    tmp_7 = split[1]
    tmp_8 = split[2]
    return tmp_6, tmp_7, tmp_8


def replacement_args(in_2):
    return (in_2, "54_7_7")


def replacement_func():
    return dispatch_coat_fusion