import torch

def pattern(x):
    return x.repeat(1, 1)

def replacement_args(x):
    return (x,)

def optimize_repeat(x):
    return x

def replacement_func():
    return optimize_repeat