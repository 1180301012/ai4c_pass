import torch
from pass_dir.dnlnet_phi_conv_mean_shared import replacement_func


def pattern(in_0, in_1, in_2, in_3):
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d.view(12, 256, -1)
    tmp_4 = in_2.mean(dim=-2, keepdim=True)
    return (tmp_4, tmp_3)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)