import torch

def pattern(x, y):
    """Simple addition pattern for reference"""
    return x + y

def replacement_args(x, y):
    """Match arguments"""
    return (x, y)

def replacement_func():
    """Return reference implementation"""
    def simple_add(x, y):
        return x + y
    return simple_add