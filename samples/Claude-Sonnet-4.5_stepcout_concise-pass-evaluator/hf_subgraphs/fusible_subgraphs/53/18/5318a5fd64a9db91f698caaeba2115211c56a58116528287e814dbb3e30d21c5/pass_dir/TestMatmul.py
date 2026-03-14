import torch

def pattern(in_0, in_1):
    """Simple test pattern - just matmul"""
    tmp_0 = in_1 @ in_0
    return tmp_0

def replacement_args(in_0, in_1):
    return (in_0, in_1)

def identity_matmul(in_0, in_1):
    """Identity replacement - just use the default @ operator"""
    return in_1 @ in_0

def replacement_func():
    return identity_matmul