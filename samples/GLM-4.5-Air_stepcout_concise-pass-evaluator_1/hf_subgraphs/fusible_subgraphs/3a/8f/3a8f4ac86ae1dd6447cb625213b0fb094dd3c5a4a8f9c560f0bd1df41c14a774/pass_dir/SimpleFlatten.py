import torch

@torch.fx.wrap
def simple_flatten(x):
    return x.flatten(1, -1)

def pattern(x):
    # Simple pattern to test basic matching
    tmp_2 = x.flatten(1, -1)
    return tmp_2

def replacement_args(x):
    return (x,)

def replacement_func():
    return simple_flatten