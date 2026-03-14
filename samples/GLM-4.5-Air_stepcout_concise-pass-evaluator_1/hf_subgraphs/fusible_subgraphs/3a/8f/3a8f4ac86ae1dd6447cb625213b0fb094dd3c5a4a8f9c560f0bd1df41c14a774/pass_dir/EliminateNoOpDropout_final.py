import torch

@torch.fx.wrap
def identity(x):
    return x

def pattern(x):
    # Dropout with p=0.0 pattern - it's just identity
    tmp_1 = torch.nn.functional.dropout(x, 0.0, False, False)
    return tmp_1

def replacement_args(x):
    return (x,)

def replacement_func():
    return identity