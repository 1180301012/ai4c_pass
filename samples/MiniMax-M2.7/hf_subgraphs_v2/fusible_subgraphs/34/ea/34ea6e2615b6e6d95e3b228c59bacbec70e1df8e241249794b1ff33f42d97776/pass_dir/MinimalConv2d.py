import torch

# Minimal pattern matching just the conv2d
def pattern(in_0, in_1, in_2, in_3, in_4):
    return torch.conv2d(in_4, in_3, in_2, (1, 1), (1, 1), (1, 1), 768)

def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_4, in_3, in_2)

def replacement_func():
    def simple_conv2d(inp, weight, bias):
        return torch.conv2d(inp, weight, bias, (1, 1), (1, 1), (1, 1), 768)
    return simple_conv2d