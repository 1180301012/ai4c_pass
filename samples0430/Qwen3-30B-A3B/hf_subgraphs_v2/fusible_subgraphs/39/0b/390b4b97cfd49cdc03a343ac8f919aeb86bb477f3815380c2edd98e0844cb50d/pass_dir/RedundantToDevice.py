import torch
from torch import device

def pattern(x):
    y = x.to(device(type='cuda', index=0))
    return y

def replacement_args(x):
    return (x,)

def remove_to_device(x):
    return x

def replacement_func():
    return remove_to_device