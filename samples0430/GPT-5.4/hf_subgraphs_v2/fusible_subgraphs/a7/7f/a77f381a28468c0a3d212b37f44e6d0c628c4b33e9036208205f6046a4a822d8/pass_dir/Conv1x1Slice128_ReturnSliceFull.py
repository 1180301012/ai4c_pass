import torch
import triton
import triton.language as tl

from pass_dir.shared_conv1x1_slice import shared_conv1x1_slice_dispatch


def pattern(in_0, in_1):
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_2 = conv2d[(slice(None, None, None), slice(None, 128, None), slice(None, None, None), slice(None, None, None))]
    return (tmp_2, conv2d)


def replacement_args(in_0, in_1):
    return (in_0, in_1, 128, (1, 1), 0)


def replacement_func():
    return shared_conv1x1_slice_dispatch