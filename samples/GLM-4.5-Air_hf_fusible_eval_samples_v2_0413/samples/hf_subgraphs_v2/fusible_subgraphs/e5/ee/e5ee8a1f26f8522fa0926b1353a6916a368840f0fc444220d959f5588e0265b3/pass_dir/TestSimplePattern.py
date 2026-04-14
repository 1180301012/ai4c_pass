import torch

def pattern(in_0, in_1, in_2):
    """Simple test pattern"""
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (1, 1), (1, 1), 128)
    return conv2d

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

def simple_replacement():
    def simple_func(in_0, in_1, in_2):
        return torch.conv2d(in_2, in_1, in_0, (1, 1), (1, 1), (1, 1), 128)
    return simple_func

def replacement_func():
    return simple_replacement