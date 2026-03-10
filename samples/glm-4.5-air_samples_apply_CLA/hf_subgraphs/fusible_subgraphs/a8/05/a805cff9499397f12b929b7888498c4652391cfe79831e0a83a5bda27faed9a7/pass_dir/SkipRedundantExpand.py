import torch
import torch.fx

@torch.fx.wrap
def identity_expand(x):
    # Redundant expand is equivalent to identity operation
    return x

def pattern(x):
    # The expand operation from [1, 1, 768] to [1, -1, -1] is redundant
    # since the result will still be [1, 1, 768]
    return x.expand(1, -1, -1)

def replacement_args(x):
    return (x,)

def replacement_func():
    return identity_expand