import torch

def pattern(x):
    return torch.nn.functional.dropout(x, 0.0, False, False)

def replacement_args(x):
    return (x,)

@torch.fx.wrap
def bypass_dropout(x):
    return x

def replacement_func():
    return bypass_dropout