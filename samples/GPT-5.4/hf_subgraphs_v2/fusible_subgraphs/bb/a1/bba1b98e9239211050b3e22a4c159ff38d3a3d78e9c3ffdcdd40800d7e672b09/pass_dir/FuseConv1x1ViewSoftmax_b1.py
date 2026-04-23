import torch
import triton
import triton.language as tl
from pass_dir.shared_conv1x1_view_softmax import fused_conv1x1_view_softmax


def pattern(conv2d):
    tmp_3 = conv2d.view(1, 1, -1)
    tmp_4 = tmp_3.softmax(dim=-1)
    return tmp_4


def replacement_args(conv2d):
    return (conv2d,)


def replacement_func():
    return fused_conv1x1_view_softmax