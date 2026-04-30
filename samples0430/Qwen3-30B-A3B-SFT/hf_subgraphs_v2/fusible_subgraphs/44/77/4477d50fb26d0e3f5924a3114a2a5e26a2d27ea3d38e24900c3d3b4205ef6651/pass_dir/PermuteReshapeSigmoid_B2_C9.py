import torch
from pass_dir.shared_permute_sigmoid import permute_reshape_sigmoid


def pattern(x):
    p = x.permute(0, 2, 3, 1)
    r = p.reshape(2, -1, 9)
    s = torch.nn.functional.sigmoid(r)
    return s


def replacement_args(x):
    return (x,)


def replacement_func():
    return permute_reshape_sigmoid