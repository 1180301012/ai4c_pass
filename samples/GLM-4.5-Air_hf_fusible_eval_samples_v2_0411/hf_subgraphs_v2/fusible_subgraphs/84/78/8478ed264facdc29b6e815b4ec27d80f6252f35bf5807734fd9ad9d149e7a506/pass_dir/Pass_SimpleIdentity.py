import torch

def pattern(x, y):
    """Pattern that matches the first operation in models: torch.matmul"""
    # This matches the first operation: tmp_0 = torch.matmul(in_0, in_1)
    tmp_0 = torch.matmul(x, y)
    return tmp_0

def replacement_args(x, y):
    return (x, y)

@torch.fx.wrap
def identity_function(x, y):
    """Identity function that returns first input tensor"""
    return x

def replacement_func():
    return identity_function