import torch

# Simple test pattern to check if multiple passes load
def pattern(x):
    return x * x

def replacement_args(x):
    return (x,)

def replacement_func():
    def simple_square(x):
        return x * x
    return simple_square