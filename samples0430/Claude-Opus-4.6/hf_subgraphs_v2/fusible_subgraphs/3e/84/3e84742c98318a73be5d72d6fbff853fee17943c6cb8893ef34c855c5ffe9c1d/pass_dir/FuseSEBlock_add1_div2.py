import torch
from pass_dir.shared_kernel import fused_se_block


def pattern(in_0, in_1, in_2, in_3):
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d + 1.0
    tmp_4 = tmp_3 / 2.0
    tmp_5 = tmp_4.clamp_(0.0, 1.0)
    tmp_6 = in_2 * tmp_5
    return (tmp_6,)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3, 1.0, 2.0)


def replacement_func():
    return fused_se_block