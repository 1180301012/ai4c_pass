import torch
from torch import device
import triton
import triton.language as tl

def pattern(in_1):
    tmp_1 = in_1.to(device(type='cuda', index=0))
    return tmp_1

def replacement_args(in_1):
    return (in_1,)

@torch.fx.wrap
def remove_device_transfer(x):
    # Simply return the input since it's already on the correct device
    return x

def replacement_func():
    return remove_device_transfer