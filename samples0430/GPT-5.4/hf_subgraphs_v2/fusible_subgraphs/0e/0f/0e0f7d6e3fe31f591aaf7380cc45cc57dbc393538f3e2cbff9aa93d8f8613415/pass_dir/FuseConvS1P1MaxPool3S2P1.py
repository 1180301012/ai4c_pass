import torch
import triton
import triton.language as tl

from pass_dir.conv_pool_shared import conv_pool_cached_dispatch


def pattern(in_0, in_1):
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (1, 1), (1, 1), 1)
    tmp_2 = torch.nn.functional.max_pool2d(conv2d, 3, 2, 1, 1, ceil_mode=False, return_indices=False)
    return tmp_2


def replacement_args(in_0, in_1):
    return (in_0, in_1, "conv_s1_p1_k3_pool3_s2_p1")


def replacement_func():
    return conv_pool_cached_dispatch