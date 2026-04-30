import torch
from pass_dir._shared import shared_dispatch_wrapper


def pattern(in_3):
    tmp_5 = in_3.sum(dim = 3, keepdim = True)
    tmp_6 = in_3 / tmp_5
    return tmp_6


def replacement_args(in_3):
    return (in_3, "sum_div_dim3")


def replacement_func():
    return shared_dispatch_wrapper