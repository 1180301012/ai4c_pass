import torch

def pattern(x):
    return torch.nn.functional.interpolate(x, (40,40), None, 'nearest')

def replacement_args(x):
    return (x,)

@torch.fx.wrap
def no_op(x):
    return x

def replacement_func():
    return no_op