import torch

def pattern(x):
    return torch.nn.functional.dropout(x, 0.1, False, False)

def replacement_args(x):
    return (x,)

def remove_dropout(x):
    return x

def replacement_func():
    return remove_dropout