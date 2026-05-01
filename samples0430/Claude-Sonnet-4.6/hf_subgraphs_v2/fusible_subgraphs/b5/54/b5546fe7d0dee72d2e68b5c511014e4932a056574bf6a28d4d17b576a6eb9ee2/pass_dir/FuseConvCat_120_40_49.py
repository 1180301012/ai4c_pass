"""coat_lite_tiny stage-4: groups=120, C_head=40, N=49, W=7"""
import torch
from pass_dir.coat_dispatch import coat_dispatch

def pattern(in_5, in_1, in_0, in_2, in_3, in_6):
    conv_out = torch.conv2d(in_5, in_1, in_0, (1, 1), (3, 3), (1, 1), 120)
    tmp_3    = torch.cat([in_2, in_3, conv_out], dim=1)
    tmp_4    = tmp_3.reshape(1, 8, 40, 49)
    tmp_5    = tmp_4.transpose(-1, -2)
    return in_6 * tmp_5

def replacement_args(in_5, in_1, in_0, in_2, in_3, in_6):
    return (in_5, in_1, in_0, in_2, in_3, in_6, "120_40_49_7")

def replacement_func():
    return coat_dispatch