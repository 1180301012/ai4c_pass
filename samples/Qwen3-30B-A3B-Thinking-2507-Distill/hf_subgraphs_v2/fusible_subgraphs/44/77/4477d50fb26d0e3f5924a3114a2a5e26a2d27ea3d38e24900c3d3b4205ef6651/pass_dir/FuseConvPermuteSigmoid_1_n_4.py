import torch
from pass_dir.shared_conv_sigmoid import fused_conv_permute_sigmoid


def pattern(in_0, in_1, in_2):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d.permute(0, 2, 3, 1)
    tmp_4 = tmp_3.reshape(1, -1, 4)
    tmp_5 = torch.sigmoid(tmp_4)
    return (tmp_5,)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, "r1_n_4")


def replacement_func():
    return fused_conv_permute_sigmoid