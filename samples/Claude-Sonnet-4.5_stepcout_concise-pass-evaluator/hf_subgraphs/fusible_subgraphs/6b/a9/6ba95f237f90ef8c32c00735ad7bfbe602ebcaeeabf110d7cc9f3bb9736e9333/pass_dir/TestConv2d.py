import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """
    Pattern to match: conv2d only
    """
    tmp_1 = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    return (tmp_1,)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@torch.fx.wrap
def passthrough_conv2d(weight, input_tensor):
    """
    Just a passthrough for now to test matching
    """
    return torch.conv2d(input_tensor, weight, None, (1, 1), (0, 0), (1, 1), 1)

def replacement_func():
    return passthrough_conv2d