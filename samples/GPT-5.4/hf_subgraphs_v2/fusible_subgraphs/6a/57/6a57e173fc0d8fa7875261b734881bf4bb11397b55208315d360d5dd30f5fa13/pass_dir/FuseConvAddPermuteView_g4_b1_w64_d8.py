import operator
import torch
from pass_dir.shared_fused_nystrom_conv import shared_replacement


def pattern(in_0, in_1, in_2):
    conv2d = torch.conv2d(in_2, in_0, None, (1, 1), (32, 0), (1, 1), 4)
    in_3 = operator.iadd(in_1, conv2d)
    tmp_3 = in_3.permute(0, 2, 1, 3)
    tmp_4 = tmp_3.contiguous()
    tmp_5 = tmp_4.view(1, 64, 32)
    return (tmp_5,)


def replacement_args(in_0, in_1, in_2):
    return (in_2, in_1, in_0, (1, 64, 32))


def replacement_func():
    return shared_replacement