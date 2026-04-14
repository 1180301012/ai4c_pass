import torch

def pattern(x, y):
    return x+y

def replacement_args(x, y):
    return (x, y)

def replacement_func():
    return lambda x, y: x + y