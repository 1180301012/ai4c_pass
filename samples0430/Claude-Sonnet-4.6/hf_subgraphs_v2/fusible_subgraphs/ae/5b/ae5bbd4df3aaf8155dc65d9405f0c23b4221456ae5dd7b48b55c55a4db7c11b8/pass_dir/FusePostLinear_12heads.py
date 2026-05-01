import torch
from pass_dir._kernels_shared import fuse_dispatch


def pattern(linear, in_2):
    """Fallback pattern: post-linear ops only, H=12."""
    tmp_4 = linear.view(1, 12, 199, 2, 4)
    tmp_5 = tmp_4.sum(-1, keepdim=False)
    tmp_6 = torch.sigmoid(tmp_5)
    chunk = tmp_6.chunk(2, dim=-1)
    tmp_8 = chunk[0]
    tmp_9 = chunk[1]
    tmp_10 = tmp_9 * in_2
    tmp_11 = tmp_10 - 1.0
    tmp_12 = tmp_8 * tmp_11
    tmp_13 = tmp_12 + 2.0
    tmp_14 = tmp_13.view(1, 12, -1, 1)
    return tmp_14


def replacement_args(linear, in_2):
    # 5-arg interface: arg3/arg4 are dummies (linear reused), route = "h12_post"
    return (linear, in_2, linear, linear, "h12_post")


def replacement_func():
    return fuse_dispatch