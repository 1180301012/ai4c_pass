import torch
from pass_dir.shared_dispatch import replacement_impl


def pattern(x):
    out = torch.nn.functional.max_pool2d(x, 2, 1, 0, 1, ceil_mode=True, return_indices=False)
    return out


def replacement_args(x):
    return (x, "maxpool2x2s1")


def replacement_func():
    return replacement_impl