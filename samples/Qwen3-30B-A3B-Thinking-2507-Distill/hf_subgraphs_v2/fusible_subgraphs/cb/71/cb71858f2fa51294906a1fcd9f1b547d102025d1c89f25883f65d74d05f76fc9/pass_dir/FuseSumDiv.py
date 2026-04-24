import torch
from pass_dir.kernels import shared_dispatch


def pattern(in_3):
    """
    Matches:
        tmp_5 = in_3.sum(dim=3, keepdim=True)
        tmp_6 = in_3 / tmp_5
    """
    tmp_5 = in_3.sum(dim=3, keepdim=True)
    tmp_6 = in_3 / tmp_5
    return tmp_6


def replacement_args(in_3):
    # Pass in_3 as a, b, c (all three slots) + route string
    # shared_dispatch uses only 'a' for the sum_div path
    return (in_3, in_3, in_3, "sum_div")


def replacement_func():
    return shared_dispatch