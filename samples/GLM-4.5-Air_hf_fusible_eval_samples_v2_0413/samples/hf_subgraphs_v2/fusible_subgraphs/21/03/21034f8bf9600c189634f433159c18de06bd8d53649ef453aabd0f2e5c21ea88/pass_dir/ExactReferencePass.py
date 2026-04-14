import torch

# Exact copy of reference pattern structure
def pattern(x, y):
    return x+y

def replacement_args(x, y):
    return (x, y)

def replacement_func():
    pass