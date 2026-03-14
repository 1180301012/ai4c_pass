import torch


def pattern(x):
    """Match dropout with p=0.0 which is a no-op"""
    return torch.nn.functional.dropout(x, 0.0, False, False)


def replacement_args(x):
    return (x,)


def replacement_func():
    return lambda x: x