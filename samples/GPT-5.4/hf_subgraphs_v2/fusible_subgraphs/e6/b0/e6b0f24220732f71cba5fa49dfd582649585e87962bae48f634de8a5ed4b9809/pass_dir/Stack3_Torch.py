import torch
from pass_dir.shared_dispatch_ops import replacement_func


def pattern(a, b, c):
    out = torch.stack([a, b, c])
    return out


def replacement_args(a, b, c):
    return (a, b, c, 'stack3')