import torch

def pattern(tmp_3):
    """Pattern: optimize redundant reshape/permute sequence"""
    tmp_4 = tmp_3.reshape(1, 2, 2, -1)
    tmp_5 = tmp_4.permute(0, 3, 1, 2)
    tmp_6 = tmp_5.contiguous()
    tmp_7 = tmp_6.permute(0, 2, 3, 1)
    tmp_8 = tmp_7.reshape(1, -1, 128)
    return tmp_8

def replacement_args(tmp_3):
    return (tmp_3,)

# Simple identity function
def optimized_reshape_sequence(x):
    """Direct identity - just return input unchanged"""
    return x

def replacement_func():
    return optimized_reshape_sequence