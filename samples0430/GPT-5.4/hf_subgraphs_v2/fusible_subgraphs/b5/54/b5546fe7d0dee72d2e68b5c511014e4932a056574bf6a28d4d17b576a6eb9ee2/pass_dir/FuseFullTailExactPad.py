import torch
from pass_dir.shared_factoratt_tail import shared_replacement_func


def pattern(tmp_4, in_4, scale, in_6, dim1, dim2):
    tmp_5 = tmp_4.transpose(-1, -2)
    tmp_6 = in_6 * tmp_5
    tmp_7 = torch._C._nn.pad(tmp_6, (0, 0, 1, 0, 0, 0), 'constant', None)
    tmp_8 = scale * in_4
    tmp_9 = tmp_8 + tmp_7
    tmp_10 = tmp_9.transpose(1, 2)
    tmp_11 = tmp_10.reshape(1, dim1, dim2)
    return tmp_11


def replacement_args(tmp_4, in_4, scale, in_6, dim1, dim2):
    return (tmp_4, in_4, scale, in_6, dim1, dim2, 'full_tail')


def replacement_func():
    return shared_replacement_func()