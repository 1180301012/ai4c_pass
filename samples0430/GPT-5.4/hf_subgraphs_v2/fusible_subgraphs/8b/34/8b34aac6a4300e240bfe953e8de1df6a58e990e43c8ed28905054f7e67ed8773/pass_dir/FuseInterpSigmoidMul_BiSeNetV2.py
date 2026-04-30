import torch
from pass_dir.shared_bisenetv2_postconv_kernels import shared_dispatch


def pattern(in_3, in_4):
    tmp_3 = torch.nn.functional.interpolate(in_4, (64, 64), None, 'bilinear', False)
    tmp_4 = torch.sigmoid(tmp_3)
    tmp_5 = in_3 * tmp_4
    return tmp_5


def replacement_args(in_3, in_4):
    return (in_3, in_4, 'branch_a')


def replacement_func():
    return shared_dispatch