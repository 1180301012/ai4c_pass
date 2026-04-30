import torch
from pass_dir.triton_conv_pool_kernels import fused_conv_pool


def pattern(in_0, in_1):
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (1, 1), (1, 1), 1)
    tmp_2 = torch.nn.functional.max_pool2d(conv2d, 3, 2, 1, 1, ceil_mode=False, return_indices=False)
    return tmp_2


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return fused_conv_pool