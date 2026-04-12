import torch

# Very simple pattern just to test if basic matching works
def pattern(x):
    return torch.nn.functional.dropout(x, p=0.0, training=False)

def replacement_args(x):
    return (x,)

# Simple identity function
@torch.fx.wrap
def identity(x):
    return x

def replacement_func():
    return identity