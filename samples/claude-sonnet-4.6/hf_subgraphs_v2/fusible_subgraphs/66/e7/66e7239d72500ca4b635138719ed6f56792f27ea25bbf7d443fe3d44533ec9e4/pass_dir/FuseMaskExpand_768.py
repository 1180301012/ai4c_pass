import torch
from pass_dir._shared import shared_dispatch


def pattern(in_0, tmp_4):
    """Match: in_0.unsqueeze(-1).expand_as(tmp_4).float() → single output tmp_7."""
    tmp_5 = in_0.unsqueeze(-1)
    tmp_6 = tmp_5.expand_as(tmp_4)
    tmp_7 = tmp_6.float()
    return tmp_7


def replacement_args(in_0, tmp_4):
    return (in_0, tmp_4, "mask_expand")


def replacement_func():
    return shared_dispatch