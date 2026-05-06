import torch
from pass_dir.gla_impl import fused_conv1x1_permute_reshape_sigmoid


def pattern(x):
    t = x.permute(0, 2, 3, 1)
    r = t.reshape(12, -1, 9)
    return (torch.nn.functional.sigmoid(r),)


def replacement_args(x):
    return (x,)


def replacement_func():
    return fused_conv1x1_permute_reshape_sigmoid