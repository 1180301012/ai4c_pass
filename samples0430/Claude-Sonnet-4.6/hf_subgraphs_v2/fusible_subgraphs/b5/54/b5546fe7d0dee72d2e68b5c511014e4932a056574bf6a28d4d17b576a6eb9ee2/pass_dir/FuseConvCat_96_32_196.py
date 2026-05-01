"""CoAT coat_lite_tiny stage-3: groups=96, C_head=32, N=196, W=14"""
import torch
from pass_dir.coat_dispatch import coat_dispatch

def pattern(in_5, in_1, in_0, in_2, in_3, in_6):
    conv_out = torch.conv2d(in_5, in_1, in_0, (1, 1), (3, 3), (1, 1), 96)
    tmp_3    = torch.cat([in_2, in_3, conv_out], dim=1)
    tmp_4    = tmp_3.reshape(1, 8, 32, 196)
    tmp_5    = tmp_4.transpose(-1, -2)
    return in_6 * tmp_5

def replacement_args(in_5, in_1, in_0, in_2, in_3, in_6):
    return (in_5, in_1, in_0, in_2, in_3, in_6, "96_32_196_14")

def replacement_func():
    return coat_dispatch