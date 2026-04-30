import torch
from pass_dir.shared_conv1x1_permute_reshape_sigmoid import fused_conv1x1_permute_reshape_sigmoid


def pattern(in_0, in_1, in_2):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d.permute(0, 2, 3, 1)
    tmp_4 = tmp_3.reshape(2, -1, 1)
    tmp_5 = torch.nn.functional.sigmoid(tmp_4)
    return tmp_5


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return fused_conv1x1_permute_reshape_sigmoid