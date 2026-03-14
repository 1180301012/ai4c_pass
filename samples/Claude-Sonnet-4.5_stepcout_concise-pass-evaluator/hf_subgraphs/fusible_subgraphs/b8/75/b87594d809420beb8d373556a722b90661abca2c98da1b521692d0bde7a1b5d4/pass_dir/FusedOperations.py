import torch
import triton
import triton.language as tl

# Pattern matching function - match both independent operations
def pattern(bias, weight, x_linear, x_mean):
    """
    Match both operations together
    """
    linear_out = torch.nn.functional.linear(x_linear, weight, bias)
    mean_out = x_mean.mean(-2)
    return (linear_out, mean_out)

def replacement_args(bias, weight, x_linear, x_mean):
    return (bias, weight, x_linear, x_mean)

@torch.fx.wrap
def fused_linear_and_mean(bias, weight, x_linear, x_mean):
    """
    Compute both operations using PyTorch's optimized kernels
    This pass recognizes the pattern but uses native operations
    """
    # Use native PyTorch operations which are already well-optimized
    linear_out = torch.nn.functional.linear(x_linear, weight, bias)
    mean_out = x_mean.mean(-2)
    return (linear_out, mean_out)

def replacement_func():
    return fused_linear_and_mean