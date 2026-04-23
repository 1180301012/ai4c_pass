import torch
from pass_dir.xclip_shared_dispatch import xclip_dispatch


def pattern(in_2):
    tmp_3 = in_2.norm(p=2, dim=-1, keepdim=True)
    tmp_4 = in_2 / tmp_3
    return tmp_4


def replacement_args(in_2):
    return (in_2, "norm_any")


def replacement_func():
    return xclip_dispatch