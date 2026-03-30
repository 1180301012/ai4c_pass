import torch

def pattern(in_0):
    return in_0

def replacement_args(in_0):
    return (in_0,)

@torch.fx.wrap
def identity_function(in_0):
    return in_0

def replacement_func():
    return identity_function