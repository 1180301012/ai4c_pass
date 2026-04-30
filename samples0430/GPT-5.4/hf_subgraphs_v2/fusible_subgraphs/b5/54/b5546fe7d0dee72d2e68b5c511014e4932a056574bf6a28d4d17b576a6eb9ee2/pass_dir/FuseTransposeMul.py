import torch
from pass_dir.shared_factoratt_tail import shared_replacement_func


def pattern(tmp_4, in_6):
    tmp_5 = tmp_4.transpose(-1, -2)
    tmp_6 = in_6 * tmp_5
    return tmp_6


def replacement_args(tmp_4, in_6):
    return (tmp_4, in_6, 'transpose_mul')


def replacement_func():
    return shared_replacement_func()