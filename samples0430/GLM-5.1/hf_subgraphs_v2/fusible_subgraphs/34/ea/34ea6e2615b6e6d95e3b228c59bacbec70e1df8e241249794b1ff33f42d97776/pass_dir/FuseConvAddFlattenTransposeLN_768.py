import torch
import triton
import triton.language as tl
from pass_dir.shared_impl import fused_impl


def pattern(in_0, in_1, in_2, in_3, in_4):
    conv2d = torch.conv2d(in_4, in_3, in_2, (1, 1), (1, 1), (1, 1), 768)
    tmp_5 = conv2d + in_4
    tmp_6 = tmp_5.flatten(2)
    tmp_7 = tmp_6.transpose(1, 2)
    tmp_8 = torch.nn.functional.layer_norm(tmp_7, (768,), in_1, in_0, 1e-05)
    tmp_9 = tmp_8.transpose(0, 1)
    tmp_10 = tmp_8.transpose(0, 1)
    return (tmp_7, tmp_10, tmp_9)


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4, "route_768")


def replacement_func():
    return fused_impl