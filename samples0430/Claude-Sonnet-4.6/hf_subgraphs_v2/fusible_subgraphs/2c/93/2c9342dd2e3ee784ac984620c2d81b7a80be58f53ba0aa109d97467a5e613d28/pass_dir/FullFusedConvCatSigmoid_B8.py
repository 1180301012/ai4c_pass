import torch
from pass_dir.full_fusion_kernel import full_fused_conv_cat_sigmoid


def pattern(in_0, in_1, in_2, in_3, in_4):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3  = conv2d.view(8, 1, -1)
    tmp_4  = torch.cat([in_3, in_4, tmp_3], 2)
    tmp_5  = tmp_4.sigmoid()
    tmp_6  = tmp_5 - 0.25
    tmp_7  = tmp_6 * 3.141592653589793
    return tmp_7


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


def replacement_func():
    return full_fused_conv_cat_sigmoid