import torch
from pass_dir.kernels import fused_linear_interpolate


def pattern(in_0, in_1, in_2):
    tmp_2 = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = tmp_2.permute(0, 2, 1)
    tmp_4 = tmp_3.reshape(24, -1, 16, 16)
    tmp_5 = torch.nn.functional.interpolate(tmp_4, size=(128, 128), mode='bilinear', align_corners=False)
    return tmp_5


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return fused_linear_interpolate