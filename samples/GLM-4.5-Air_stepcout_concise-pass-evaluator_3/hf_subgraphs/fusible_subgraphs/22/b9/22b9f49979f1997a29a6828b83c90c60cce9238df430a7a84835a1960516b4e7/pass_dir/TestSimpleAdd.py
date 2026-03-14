import torch

def pattern(x, y):
    """Simple addition pattern"""
    return x + y

def replacement_args(x, y):
    return (x, y)

@torch.fx.wrap  
def simple_add(x, y):
    """Simple addition that just calls torch.add"""
    return torch.add(x, y)

def replacement_func():
    return simple_add