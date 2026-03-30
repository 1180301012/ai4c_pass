import torch

def pattern(x, y):
    """Very basic test pattern"""
    return x + y

def replacement_args(x, y):
    return (x, y)

def simple_add(x, y):
    """Simple addition function"""
    return x + y

def replacement_func():
    return simple_add