import torch
from pass_dir.conv_depthwise_impl import depthwise_conv_add_permute


def pattern(in_0, in_1, in_2):
    conv2d = torch.conv2d(in_2, in_0, None, (1, 1), (32, 0), (1, 1), 4)
    in_1 += conv2d
    tmp_3 = in_1.permute(0, 2, 1, 3)
    tmp_4 = tmp_3.contiguous()
    tmp_5 = tmp_4.view(8, 256, 32)
    return (tmp_5,)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return depthwise_conv_add_permute