import torch
from pass_dir.fused_kernel import fused_coat_crpe_dispatch

def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    conv2d = torch.conv2d(in_5, in_1, in_0, (1, 1), (3, 3), (1, 1), 96)
    tmp_3 = torch.cat([in_2, in_3, conv2d], dim = 1)
    tmp_4 = tmp_3.reshape(1, 8, 32, 196)
    tmp_5 = tmp_4.transpose(-1, -2)
    tmp_6 = in_6 * tmp_5
    tmp_7 = torch.nn.functional.pad(tmp_6, (0, 0, 1, 0, 0, 0), 'constant', None)
    tmp_8 = 0.1767766952966369 * in_4
    tmp_9 = tmp_8 + tmp_7
    tmp_10 = tmp_9.transpose(1, 2)
    tmp_11 = tmp_10.reshape(1, 197, 256)
    return (tmp_11,)

def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6, "route_96_0.1767766952966369")

def replacement_func():
    return fused_coat_crpe_dispatch