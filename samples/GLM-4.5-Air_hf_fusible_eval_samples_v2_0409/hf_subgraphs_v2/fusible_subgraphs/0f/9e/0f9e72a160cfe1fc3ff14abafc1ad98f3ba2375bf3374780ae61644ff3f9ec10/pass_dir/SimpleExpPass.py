import torch

def pattern(in_0):
    """Pattern to match simple exponential operation"""
    tmp_5 = in_0.exp()
    return tmp_5

def replacement_args(in_0):
    return (in_0,)

def replacement_func():
    def exp_forward(x):
        return x.exp()
    return exp_forward