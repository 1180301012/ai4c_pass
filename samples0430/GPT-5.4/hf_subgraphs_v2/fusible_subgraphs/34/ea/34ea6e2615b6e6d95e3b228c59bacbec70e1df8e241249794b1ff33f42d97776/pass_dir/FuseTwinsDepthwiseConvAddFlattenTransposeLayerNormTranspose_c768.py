import torch
from pass_dir.twins_dwconv_ln_common import shared_replacement_func


def pattern(conv2d, in_4):
    tmp_5 = conv2d + in_4
    tmp_6 = tmp_5.flatten(2)
    tmp_7 = tmp_6.transpose(1, 2)
    return tmp_7


def replacement_args(conv2d, in_4):
    return (conv2d, in_4)


def replacement_func():
    return shared_replacement_func()