import torch

# Exact unoptimized example from instructions
def pattern(x, y):
    return x+y

def replacement_args(x, y):
    return (x, y)

def replacement_func():
    pass