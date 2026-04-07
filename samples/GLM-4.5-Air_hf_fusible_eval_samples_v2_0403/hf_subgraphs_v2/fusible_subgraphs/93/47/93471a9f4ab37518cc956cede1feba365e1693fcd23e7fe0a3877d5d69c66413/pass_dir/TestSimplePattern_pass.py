import torch

def pattern(a, b):
    # Exact pattern from reference
    t = a
    out = b
    return t, out

def replacement_args(a, b):
    return (a, b)

def replacement_func():
    return lambda a, b: (a, b)