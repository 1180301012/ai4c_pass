import torch

def pattern(a, b):
    return a + b

def replacement_args(a, b):
    return (a, b)

def replacement_func():
    def simple_replacement(x, y):
        return x + y
    return simple_replacement