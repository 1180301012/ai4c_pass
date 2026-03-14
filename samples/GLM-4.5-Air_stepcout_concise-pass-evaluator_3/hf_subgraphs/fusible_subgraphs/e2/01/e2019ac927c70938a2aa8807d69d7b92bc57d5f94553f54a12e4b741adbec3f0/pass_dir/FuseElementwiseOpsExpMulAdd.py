import torch

def pattern(in_1, in_2, in_0):
    """Fused pattern: exp(in_1) * in_2 + in_0"""
    tmp_0 = in_1.exp()
    tmp_1 = in_2 * tmp_0
    tmp_2 = tmp_1 + in_0
    return tmp_2

def replacement_args(in_1, in_2, in_0):
    return (in_1, in_2, in_0)

def fused_elementwise_ops(in_1, in_2, in_0):
    # Minimal overhead implementation for small tensors
    return in_2 * in_1.exp() + in_0

def replacement_func():
    return fused_elementwise_ops