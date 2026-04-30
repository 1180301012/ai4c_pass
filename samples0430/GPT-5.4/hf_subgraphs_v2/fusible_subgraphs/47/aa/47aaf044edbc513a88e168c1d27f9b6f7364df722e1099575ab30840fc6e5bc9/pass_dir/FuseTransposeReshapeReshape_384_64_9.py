import torch
from pass_dir.convbert_shared import dispatch_convbert_replacement


def pattern(in_0):
    tmp_0 = in_0.transpose(1, 2)
    tmp_1 = tmp_0.reshape(1, -1, 384, 9)
    tmp_2 = torch.reshape(tmp_1, [-1, 64, 9])
    return tmp_2


def replacement_args(in_0):
    return (in_0, "pack_384_64")


def replacement_func():
    return dispatch_convbert_replacement