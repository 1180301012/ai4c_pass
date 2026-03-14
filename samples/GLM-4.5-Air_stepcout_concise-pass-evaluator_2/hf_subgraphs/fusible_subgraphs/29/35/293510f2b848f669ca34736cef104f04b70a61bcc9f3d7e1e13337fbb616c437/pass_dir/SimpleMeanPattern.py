import torch

def pattern(in_0):
    """Just match a simple mean operation"""
    tmp_1 = in_0.mean((2, 3), keepdim=True)
    return tmp_1

def replacement_args(in_0):
    return (in_0,)

@torch.fx.wrap
def simple_mean(x):
    """Simple mean replacement"""
    return x.mean((2, 3), keepdim=True)

def replacement_func():
    return simple_mean