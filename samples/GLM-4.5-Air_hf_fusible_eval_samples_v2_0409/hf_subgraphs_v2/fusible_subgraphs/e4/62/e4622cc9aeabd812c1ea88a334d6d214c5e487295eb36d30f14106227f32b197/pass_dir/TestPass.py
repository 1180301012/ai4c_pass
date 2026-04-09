import torch

def pattern(x):
    return x + 1

def replacement_args(x):
    return (x,)

@torch.fx.wrap
def test_function(x):
    return x

def replacement_func():
    return test_function