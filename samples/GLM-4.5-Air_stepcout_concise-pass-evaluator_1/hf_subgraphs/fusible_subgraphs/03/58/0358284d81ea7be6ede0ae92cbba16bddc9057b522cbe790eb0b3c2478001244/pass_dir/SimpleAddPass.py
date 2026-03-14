import torch

def pattern(x, y):
    """Very simple pattern to test framework"""
    return x + y

def replacement_args(x, y):
    return (x, y)

def replacement_func():
    # Just return a simple add function for now
    def add_func(x, y):
        return x + y
    return add_func