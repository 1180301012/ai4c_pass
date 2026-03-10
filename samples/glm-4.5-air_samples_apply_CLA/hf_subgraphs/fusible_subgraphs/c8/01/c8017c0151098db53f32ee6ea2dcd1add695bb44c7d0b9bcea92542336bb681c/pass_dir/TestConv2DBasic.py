import torch

# Simple pattern: Just conv2d to test matching
def pattern(input_tensor, weight, bias):
    return torch.conv2d(input_tensor, weight, bias, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1)

# Extract arguments
def replacement_args(input_tensor, weight, bias):
    return (input_tensor, weight, bias)

# Simple replacement - just use exact same function
def replacement_func():
    return pattern