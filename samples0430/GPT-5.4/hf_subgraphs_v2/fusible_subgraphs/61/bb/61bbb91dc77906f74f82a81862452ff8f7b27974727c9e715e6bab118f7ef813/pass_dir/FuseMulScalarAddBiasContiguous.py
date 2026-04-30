import torch
from pass_dir.shared_ccnet_kernels import shared_replacement_func


def pattern(in_5, in_0, in_2):
    tmp_3 = in_5 * in_0
    tmp_4 = tmp_3 + in_2
    tmp_5 = tmp_4.contiguous()
    return tmp_5


def replacement_args(in_5, in_0, in_2):
    return (in_5, in_0, in_2, 'epilogue')


def replacement_func():
    return shared_replacement_func()