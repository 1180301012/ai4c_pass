import torch

# Simple addition pattern for understanding the framework
def pattern(x, y):
    return x + y

def replacement_args(x, y):
    return (x, y)

@torch.fx.wrap
def simple_add(x, y):
    return x + y

def replacement_func():
    return simple_add