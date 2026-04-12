import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    """
    Test pattern: just match the linear transformation
    """
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    return linear

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2)

def replacement_func():
    # Simple placeholder
    def simple_linear(in_0, in_1, in_2):
        return torch.nn.functional.linear(in_2, in_1, in_0)
    return simple_linear