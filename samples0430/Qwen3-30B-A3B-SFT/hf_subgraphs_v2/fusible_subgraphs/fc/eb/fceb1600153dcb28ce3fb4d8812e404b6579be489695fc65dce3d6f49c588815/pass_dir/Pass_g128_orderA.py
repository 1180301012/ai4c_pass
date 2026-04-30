import torch
from pass_dir.kernel_conv_bn_mean import fused_conv_bn_mean


def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    conv2d = torch.conv2d(in_7, in_5, in_4, (1, 1), (0, 0), (1, 1), 128)
    tmp_7 = in_6 + conv2d
    tmp_8 = tmp_7 + in_7
    tmp_9 = torch.nn.functional.batch_norm(tmp_8, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_10 = tmp_9.mean((2, 3), keepdim=True)
    return (tmp_9, tmp_10)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, 128)


def replacement_func():
    return fused_conv_bn_mean