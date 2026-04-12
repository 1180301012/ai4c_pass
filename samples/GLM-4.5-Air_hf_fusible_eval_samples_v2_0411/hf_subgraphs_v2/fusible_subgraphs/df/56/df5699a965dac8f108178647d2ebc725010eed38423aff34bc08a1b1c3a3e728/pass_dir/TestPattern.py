import torch

# Simple test pattern to understand matching
def pattern(x):
    return x + x

def replacement_args(x):
    return (x,)

def replacement_func():
    def simple_add(x):
        return x + x
    return simple_add