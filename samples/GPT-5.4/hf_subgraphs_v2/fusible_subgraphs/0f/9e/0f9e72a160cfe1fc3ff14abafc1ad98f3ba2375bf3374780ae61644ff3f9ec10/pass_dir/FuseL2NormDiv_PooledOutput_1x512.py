import torch
from pass_dir.xclip_shared_dispatch import xclip_dispatch


def pattern(in_1):
    tmp_1 = in_1.norm(p=2, dim=-1, keepdim=True)
    tmp_2 = in_1 / tmp_1
    return tmp_2


def replacement_args(in_1):
    return (in_1, "norm_any")


def replacement_func():
    return xclip_dispatch