import torch
from pass_dir.shared_dmnet_tail import shared_dmnet_dispatch


def pattern(in_5, in_6, in_7, in_8, tmp_7):
    tmp_8 = torch.cat([in_5, in_7, in_8, in_6, tmp_7], dim=1)
    return tmp_8


def replacement_args(in_5, in_6, in_7, in_8, tmp_7):
    return (in_5, in_6, in_7, in_8, tmp_7, "cat5")


def replacement_func():
    return shared_dmnet_dispatch