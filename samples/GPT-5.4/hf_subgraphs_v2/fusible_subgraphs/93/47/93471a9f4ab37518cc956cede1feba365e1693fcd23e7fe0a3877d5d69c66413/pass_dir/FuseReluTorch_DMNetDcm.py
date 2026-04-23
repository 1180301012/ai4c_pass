import torch
from pass_dir.shared_dmnet_tail import shared_dmnet_dispatch


def pattern(x):
    y = torch.relu(x)
    return y


def replacement_args(x):
    return (x, "relu")


def replacement_func():
    return shared_dmnet_dispatch