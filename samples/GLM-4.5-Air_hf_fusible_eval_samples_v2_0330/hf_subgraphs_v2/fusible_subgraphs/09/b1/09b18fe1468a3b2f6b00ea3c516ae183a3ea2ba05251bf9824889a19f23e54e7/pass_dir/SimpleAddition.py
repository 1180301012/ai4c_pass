import torch

@torch.fx.wrap
def simple_add(x, y):
    """Optimized subtraction using PyTorch's native operations"""
    # The subtraction operation with broadcasting is already optimized in PyTorch
    # This pass demonstrates that the pattern matching works
    return x - y

def pattern(in_1, in_2):
    tmp_1 = in_1 - in_2
    return tmp_1

def replacement_args(in_1, in_2):
    return (in_1, in_2)

def replacement_func():
    return simple_add