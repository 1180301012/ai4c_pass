import torch

def pattern(x):
    return x

def replacement_args(x):
    return (x,)

def replacement_func():
    def identity(x):
        return x
    return identity