import torch

def pattern(a, b, c):
    """Match the computation pattern: matmul + scale + transpose"""
    tmp = torch.matmul(a, b)
    scaled = tmp * c
    transposed = scaled.t()
    return scaled, transposed

def replacement_args(in_0, in_1, in_2):
    """Extract arguments for the replacement function"""
    return in_0, in_1, in_2

def replacement_func():
    """Return a simple test function to see if patterns work"""
    def simple_test(a, b, c):
        tmp = torch.matmul(a, b)
        scaled = tmp * c
        transposed = scaled.t()
        return scaled, transposed
    return simple_test