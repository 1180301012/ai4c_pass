import torch

def pattern(in_0):
    tmp_1 = in_0.transpose(-2, -1)
    return tmp_1

def replacement_args(in_0):
    return (in_0,)

def optimized_transpose(in_0):
    """
    Use PyTorch's built-in transpose which is already highly optimized
    """
    return in_0.transpose(-2, -1)

def replacement_func():
    return optimized_transpose