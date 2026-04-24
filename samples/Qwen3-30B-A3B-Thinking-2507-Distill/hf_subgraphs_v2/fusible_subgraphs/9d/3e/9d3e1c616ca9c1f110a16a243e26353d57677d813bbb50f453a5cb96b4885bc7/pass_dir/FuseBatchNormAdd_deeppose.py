"""
Full forward pass: conv2d + batch_norm + in-place add (deeppose pattern).
Both this pass and FuseBatchNormAdd_resnet10t import _dispatch from shared_kernels.py
so they share the SAME Python function object, satisfying replacement_func_limit.
"""
import torch
from pass_dir.shared_kernels import _dispatch


def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    conv2d = torch.conv2d(in_6, in_4, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_6 = torch.nn.functional.batch_norm(conv2d, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_6 += in_5
    return tmp_6


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    # (conv_w=in_4, rm=in_0, rv=in_1, bw=in_3, bb=in_2, x=in_6, res=in_5, route)
    return (in_4, in_0, in_1, in_3, in_2, in_6, in_5, "deeppose")


def replacement_func():
    return _dispatch