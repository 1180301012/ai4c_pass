import torch
import triton
import triton.language as tl


def pattern(x):
    return x.unsqueeze(2)


def replacement_args(x):
    return (x,)


@torch.fx.wrap
def triton_unsqueeze(x):
    # unsqueeze is just a view operation - no computation needed
    return x.unsqueeze(2)


def replacement_func():
    return triton_unsqueeze