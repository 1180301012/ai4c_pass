import torch

def pattern(x, y):
    return x+y

def replacement_args(x, y):
    return (x, y)

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    pass