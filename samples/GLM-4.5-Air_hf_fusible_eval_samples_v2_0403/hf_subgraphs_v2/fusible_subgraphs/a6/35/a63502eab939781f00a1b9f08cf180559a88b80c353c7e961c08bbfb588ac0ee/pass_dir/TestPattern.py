import torch

def pattern(tmp_3, tmp_4):
    """Simple test pattern to verify framework is working"""
    return torch.add(tmp_3, tmp_4)

def replacement_args(tmp_3, tmp_4):
    return (tmp_3, tmp_4)

def replacement_func():
    def simple_add(a, b):
        return torch.add(a, b)
    return simple_add