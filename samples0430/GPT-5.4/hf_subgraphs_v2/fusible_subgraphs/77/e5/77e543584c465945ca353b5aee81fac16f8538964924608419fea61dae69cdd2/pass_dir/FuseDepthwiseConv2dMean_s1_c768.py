import torch
import triton
import triton.language as tl
from pass_dir.depthwise_conv_mean_shared import dispatch_depthwise_conv_mean


def pattern(in_0, in_1):
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (1, 1), (1, 1), 768)
    tmp_2 = conv2d.mean((2, 3), keepdim=True)
    return conv2d, tmp_2


def replacement_args(in_0, in_1):
    return (in_0, in_1, "s1")


def replacement_func():
    return dispatch_depthwise_conv_mean