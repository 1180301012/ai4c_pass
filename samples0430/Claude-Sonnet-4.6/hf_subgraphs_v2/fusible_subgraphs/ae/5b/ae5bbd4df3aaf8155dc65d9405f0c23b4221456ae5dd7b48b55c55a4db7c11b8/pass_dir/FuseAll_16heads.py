import torch
from pass_dir._kernels_shared import fuse_dispatch


def pattern(bias, weight, in_2, query):
    """Full pattern: linear + entire post-processing chain for H=16."""
    linear = torch.nn.functional.linear(query, weight, bias)
    tmp_4 = linear.view(1, 16, 199, 2, 4)
    tmp_5 = tmp_4.sum(-1, keepdim=False)
    tmp_6 = torch.sigmoid(tmp_5)
    chunk = tmp_6.chunk(2, dim=-1)
    tmp_8 = chunk[0]
    tmp_9 = chunk[1]
    tmp_10 = tmp_9 * in_2
    tmp_11 = tmp_10 - 1.0
    tmp_12 = tmp_8 * tmp_11
    tmp_13 = tmp_12 + 2.0
    tmp_14 = tmp_13.view(1, 16, -1, 1)
    return tmp_14


def replacement_args(bias, weight, in_2, query):
    # 5-arg interface: (bias, weight, in_2, query, route)
    return (bias, weight, in_2, query, "h16_full")


def replacement_func():
    return fuse_dispatch