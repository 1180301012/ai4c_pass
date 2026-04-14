import torch
from pass_dir.shared_kernel import auto_permute_reshape


def pattern(x):
    t = x.permute(0, 2, 3, 1)
    r = t.reshape(12, -1, 9)
    return r


def replacement_args(x):
    return (x,)


def replacement_func():
    return auto_permute_reshape