import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    # Try to match just a simple conv2d operation
    conv_out = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    # Return the conv output to see if we can match just this part
    return conv_out,

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@torch.fx.wrap
def conv2d_only_impl(weights, input_tensor):
    # Simple conv2d implementation to test matching
    out = torch.conv2d(input_tensor, weights, None, (1, 1), (0, 0), (1, 1), 1)
    return (out,)

def replacement_func():
    return conv2d_only_impl