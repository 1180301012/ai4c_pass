import torch
from pass_dir.shared_kernel import fused_crpe

def pattern(conv2d, in_2, in_3, in_4, in_6):
    tmp_3 = torch.cat([in_2, in_3, conv2d], dim=1)
    tmp_4 = tmp_3.reshape(1, 8, 32, 196)
    tmp_5 = tmp_4.transpose(-1, -2)
    tmp_6 = in_6 * tmp_5
    tmp_7 = torch._C._nn.pad(tmp_6, (0, 0, 1, 0, 0, 0), 'constant', None)
    tmp_8 = 0.1767766952966369 * in_4
    tmp_9 = tmp_8 + tmp_7
    tmp_10 = tmp_9.transpose(1, 2)
    tmp_11 = tmp_10.reshape(1, 197, 256)
    return tmp_11

def replacement_args(conv2d, in_2, in_3, in_4, in_6):
    return (conv2d, in_2, in_3, in_4, in_6)

def replacement_func():
    return fused_crpe