import torch
from pass_dir.shared_tiny_bat_resnext26ts_609_614 import shared_replacement_dispatch


# Single-output pass for row-wise normalization over dim=3.
def pattern(in_3):
    tmp_5 = in_3.sum(dim=3, keepdim=True)
    tmp_6 = in_3 / tmp_5
    return tmp_6


def replacement_args(in_3):
    return (in_3, "row_normalize")


def replacement_func():
    return shared_replacement_dispatch