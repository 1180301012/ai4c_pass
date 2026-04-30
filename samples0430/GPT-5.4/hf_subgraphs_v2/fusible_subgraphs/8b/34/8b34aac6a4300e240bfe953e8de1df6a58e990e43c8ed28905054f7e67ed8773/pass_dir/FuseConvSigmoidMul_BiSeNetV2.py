import torch
from pass_dir.shared_bisenetv2_postconv_kernels import shared_dispatch


def pattern(tmp_2, in_2):
    tmp_6 = torch.sigmoid(tmp_2)
    tmp_7 = in_2 * tmp_6
    return tmp_7


def replacement_args(tmp_2, in_2):
    return (tmp_2, in_2, 'sigmul')


def replacement_func():
    return shared_dispatch