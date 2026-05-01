import torch

def pattern(tmp_8):
    return torch.nn.functional.dropout(tmp_8, 0.0, False, False)

def replacement_args(tmp_8):
    return (tmp_8,)

@torch.fx.wrap
def identity(x):
    return x

def replacement_func():
    return identity