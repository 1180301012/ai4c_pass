import torch
from pass_dir.shared_kernel import auto_permute_reshape


def pattern(x):
    t = x.permute(0, 2, 3, 1)
    r = t.reshape(1, -1, 36)
    return r


def replacement_args(x):
    return (x,)


def replacement_func():
    return auto_permute_reshape