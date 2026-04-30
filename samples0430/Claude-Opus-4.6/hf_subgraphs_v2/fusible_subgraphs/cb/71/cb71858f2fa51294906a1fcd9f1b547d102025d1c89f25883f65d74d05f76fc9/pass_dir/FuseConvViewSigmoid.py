import torch
from pass_dir.shared_dispatch import dispatch


def pattern(input_tensor, weight, bias):
    conv2d = torch.conv2d(input_tensor, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d.view(1, 2, 8, 8)
    tmp_4 = tmp_3.sigmoid()
    return tmp_4


def replacement_args(input_tensor, weight, bias):
    return (input_tensor, weight, bias, "conv")


def replacement_func():
    return dispatch