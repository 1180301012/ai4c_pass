import torch
from pass_dir.fused_kernels import fused_conv2d_gelu_drop_dispatch


def pattern(in_0, in_1, in_2):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (1, 1), (1, 1), 256)
    tmp_3 = torch.nn.functional.gelu(conv2d)
    tmp_4 = torch.nn.functional.dropout(tmp_3, 0.0, False, False)
    return (tmp_4,)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, "dw_256")


def replacement_func():
    return fused_conv2d_gelu_drop_dispatch