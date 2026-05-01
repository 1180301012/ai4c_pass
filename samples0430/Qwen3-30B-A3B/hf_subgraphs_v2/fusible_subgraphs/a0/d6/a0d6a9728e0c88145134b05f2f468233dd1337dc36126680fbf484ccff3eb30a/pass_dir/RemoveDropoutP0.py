import torch

def pattern(x):
    return torch.nn.functional.dropout(x, p=0.0, training=False)

def replacement_args(x):
    return (x,)

def identity(x):
    return x

def replacement_func():
    return identity