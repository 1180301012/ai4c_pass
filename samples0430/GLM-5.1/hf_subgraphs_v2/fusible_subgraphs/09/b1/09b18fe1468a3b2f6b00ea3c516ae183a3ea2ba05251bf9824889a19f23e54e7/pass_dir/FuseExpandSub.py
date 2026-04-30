import torch
from pass_dir.shared_dispatch import dispatch_wrapper


def pattern(in_4, in_0):
    tmp_7 = in_4.unsqueeze(2)
    tmp_8 = tmp_7.expand((1, 4096, 32, 512))
    tmp_6 = in_0.view((1, 1, 32, 512))
    tmp_10 = tmp_8 - tmp_6
    return tmp_10


def replacement_args(in_4, in_0):
    return (in_4, in_0, "expand_sub")


def replacement_func():
    return dispatch_wrapper