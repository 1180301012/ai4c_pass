import torch
from pass_dir.shared_bisenetv2_postconv_kernels import shared_dispatch


def pattern(tmp_2, in_2, in_3, in_4):
    tmp_3 = torch.nn.functional.interpolate(in_4, (64, 64), None, 'bilinear', False)
    tmp_4 = torch.sigmoid(tmp_3)
    tmp_5 = in_3 * tmp_4
    tmp_6 = torch.sigmoid(tmp_2)
    tmp_7 = in_2 * tmp_6
    tmp_8 = torch.nn.functional.interpolate(tmp_7, (64, 64), None, 'bilinear', False)
    tmp_9 = tmp_5 + tmp_8
    return tmp_9


def replacement_args(tmp_2, in_2, in_3, in_4):
    return (tmp_2, in_2, in_3, in_4, 'full')


def replacement_func():
    return shared_dispatch