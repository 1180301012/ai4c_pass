"""
Full forward pass: conv2d + batch_norm + in-place add (resnet10t pattern).
Both this pass and FuseBatchNormAdd_deeppose import _dispatch from shared_kernels.py
so they share the SAME Python function object, satisfying replacement_func_limit.
"""
import torch
from pass_dir.shared_kernels import _dispatch


def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    """
    in_0=conv_weight, in_1=rm, in_2=rv, in_3=bn_bias, in_4=bn_weight,
    in_5=input, in_6=residual
    """
    conv2d = torch.conv2d(in_5, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_6 = torch.nn.functional.batch_norm(conv2d, in_1, in_2, in_4, in_3, False, 0.1, 1e-05)
    in_6 += tmp_6
    return in_6


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    # _dispatch signature: (conv_w=a0, rm=a1, rv=a2, bn_bias=a3, bn_weight=a4, x=a5, res=a6, route)
    # in_0=conv_weight, in_1=rm, in_2=rv, in_3=bn_bias, in_4=bn_weight, in_5=input, in_6=residual
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6, "resnet10t")


def replacement_func():
    return _dispatch