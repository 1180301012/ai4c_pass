import torch
from pass_dir.shared_dispatch import dispatch


def pattern(in_2):
    tmp_2 = in_2.transpose(-1, -2)
    return tmp_2


def replacement_args(in_2):
    return (in_2, "route_transpose")


def replacement_func():
    return dispatch