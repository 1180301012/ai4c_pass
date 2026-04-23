import torch
from pass_dir.shared_dmnet_tail import shared_dmnet_dispatch


def pattern(x):
    y = x.view(1, 512, 64, 64)
    return y


def replacement_args(x):
    return (x, "view")


def replacement_func():
    return shared_dmnet_dispatch