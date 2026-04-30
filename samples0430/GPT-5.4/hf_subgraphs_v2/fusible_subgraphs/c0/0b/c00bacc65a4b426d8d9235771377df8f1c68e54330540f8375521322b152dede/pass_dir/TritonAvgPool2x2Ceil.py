import torch
from pass_shared_impl import shared_dispatch


def pattern(x):
    return torch.nn.functional.avg_pool2d(x, 2, 2, 0, True, False, None)


def replacement_args(x):
    return (x, "avgpool2x2_ceil")


def replacement_func():
    return shared_dispatch