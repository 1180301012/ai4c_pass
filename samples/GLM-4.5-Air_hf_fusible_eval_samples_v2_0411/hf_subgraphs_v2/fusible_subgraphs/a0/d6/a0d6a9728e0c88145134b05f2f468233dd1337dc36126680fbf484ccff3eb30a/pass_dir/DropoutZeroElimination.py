import torch

def pattern(x):
    # Specifically match dropout with p=0.0
    return torch.nn.functional.dropout(x, p=0.0, training=False)

def replacement_args(x):
    return (x,)

# Identity function - dropout with p=0.0 is just the input
@torch.fx.wrap
def identity(x):
    return x

def replacement_func():
    return identity