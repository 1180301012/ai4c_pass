import torch

def pattern(in_1):
    return torch.nn.functional.dropout(in_1, 0.0, False, False)

def replacement_args(in_1):
    return (in_1,)

def no_op(x):
    return x

def replacement_func():
    return no_op