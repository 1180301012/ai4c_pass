import torch

def pattern(linear):
    return torch.nn.functional.dropout(linear, 0.0, False, False)

def replacement_args(linear):
    return (linear,)

@torch.fx.wrap
def identity(x):
    return x

def replacement_func():
    return identity