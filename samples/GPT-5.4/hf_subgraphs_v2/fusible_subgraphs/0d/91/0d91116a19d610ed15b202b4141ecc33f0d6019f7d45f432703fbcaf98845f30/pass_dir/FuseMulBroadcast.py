import torch
from pass_dir.shared_fused_bn_silu import dispatch_replacement


def pattern(in_4, in_5):
    tmp_4 = in_5 * in_4
    return (tmp_4,)


def replacement_args(in_4, in_5):
    return (in_4, in_5, "mul")


def replacement_func():
    return dispatch_replacement