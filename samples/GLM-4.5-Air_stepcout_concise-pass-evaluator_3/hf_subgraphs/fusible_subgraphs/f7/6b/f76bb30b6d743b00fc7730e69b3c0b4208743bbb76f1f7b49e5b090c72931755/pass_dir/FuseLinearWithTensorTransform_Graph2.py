import torch

def pattern(x, y):
    """Simple addition pattern for Graph 2"""
    return x + y

def replacement_args(x, y):
    """Extract arguments for addition pattern - Graph 2"""
    return (x, y)

def replacement_func():
    """Return a simple addition function for Graph 2"""
    def simple_add(x, y):
        return x + y
    return simple_add