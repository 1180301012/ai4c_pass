import torch
import torch.fx

@torch.fx.wrap
def identity_dropout(x):
    # Dropout with p=0.0 is equivalent to identity operation
    return x

def pattern(x):
    return torch.nn.functional.dropout(x, 0.0, False, False)

def replacement_args(x):
    return (x,)

def replacement_func():
    return identity_dropout