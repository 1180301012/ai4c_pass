import torch

def pattern(x):
    result = torch.nn.functional.dropout(x, 0.0, False, False)
    return result

def replacement_args(x):
    return (x,)

def identity_func(x):
    return x

def replacement_func():
    return identity_func