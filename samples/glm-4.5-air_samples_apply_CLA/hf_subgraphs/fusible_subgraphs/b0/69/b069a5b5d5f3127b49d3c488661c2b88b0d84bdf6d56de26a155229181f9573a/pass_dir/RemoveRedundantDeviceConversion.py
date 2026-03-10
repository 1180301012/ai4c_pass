import torch
from torch import device
import triton
import triton.language as tl


def pattern(x):
    tmp_1 = x.to(device(type='cuda', index=0))
    return tmp_1


def replacement_args(x):
    return (x,)


@torch.fx.wrap
def identity(x):
    return x

def replacement_func():
    return identity