import torch

def pattern(x, y):
    """Simple pattern test"""
    return x + y

def replacement_args(x, y):
    return (x, y)

def replacement_func():
    return torch.add