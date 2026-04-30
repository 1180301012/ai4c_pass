import torch
from pass_dir.shared_dispatch import replacement_func


def pattern(in_0, in_1):
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_2 = torch.sigmoid(conv2d)
    return tmp_2


def replacement_args(in_0, in_1):
    return (in_0, in_1, 'conv_sigmoid')