import torch

def pattern(x, p, train, inplace):
    if p == 0.0 and not train and not inplace:
        return torch.nn.functional.dropout(x, p, train, inplace)
    return None

def replacement_args(x, p, train, inplace):
    return (x,)

def no_op(x):
    return x

def replacement_func():
    return no_op