import torch

# Very simple pattern to test the mechanism
def pattern(in_0, in_1, in_2):
    """Simple linear pattern to test matching"""
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    return linear

# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

# Simple replacement function (just identity)
def replacement_func():
    def simple_identity(in_0, in_1, in_2):
        return torch.nn.functional.linear(in_2, in_1, in_0)
    return simple_identity