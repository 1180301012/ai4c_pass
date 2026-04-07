import torch

# Pattern matching function
def pattern(in_0, in_1, in_2):
    # Match the exact computation pattern from model.py
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d * 1.0
    tmp_4 = tmp_3.reshape(-1, 17, 4096)
    return tmp_4

# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

# Simple optimized implementation that eliminates the no-op multiplication
def optimized_simple_implementation(in_0, in_1, in_2):
    """Simply eliminates the no-op multiplication by 1.0"""
    # Perform the convolution
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    
    # Eliminate the no-op multiplication by 1.0
    # Directly reshape the convolution result
    output = conv2d.reshape(-1, 17, 4096)
    
    return output

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return optimized_simple_implementation