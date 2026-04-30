import torch
from pass_dir.shared_factoratt_tail import shared_replacement_func


def pattern(in_4, scale, tmp_7, dim1, dim2):
    tmp_8 = scale * in_4
    tmp_9 = tmp_8 + tmp_7
    tmp_10 = tmp_9.transpose(1, 2)
    tmp_11 = tmp_10.reshape(1, dim1, dim2)
    return tmp_11


def replacement_args(in_4, scale, tmp_7, dim1, dim2):
    return (in_4, scale, tmp_7, dim1, dim2, 'scale_add_transpose_reshape')


def replacement_func():
    return shared_replacement_func()