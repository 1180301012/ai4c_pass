import torch

# Very basic pattern: Try to match just conv2d with minimal parameters
def pattern(x, weight, bias):
    return torch.conv2d(x, weight, bias, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1)

# Extract arguments
def replacement_args(x, weight, bias):
    return (x, weight, bias)

# Simple replacement - just use the same function for now
def replacement_func():
    return pattern