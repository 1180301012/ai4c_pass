import torch
import triton
import triton.language as tl

def pattern(in_3, weight, bias, in_2):
    """Pattern to match: entire computation"""
    tmp_2 = torch.nn.functional.linear(in_3, weight, bias)
    tmp_3 = tmp_2.permute(0, 3, 1, 2)
    tmp_4 = in_2.transpose(-2, -1)
    return (tmp_3, tmp_4)

def replacement_args(in_3, weight, bias, in_2):
    return (in_3, weight, bias, in_2)

# Simple pass-through implementation - no optimization, just verify pattern matching
@torch.fx.wrap
def fused_computation(in_3, weight, bias, in_2):
    # Use native PyTorch operations (highly optimized)
    tmp_2 = torch.nn.functional.linear(in_3, weight, bias)
    tmp_3 = tmp_2.permute(0, 3, 1, 2)
    tmp_4 = in_2.transpose(-2, -1)
    return (tmp_3, tmp_4)

def replacement_func():
    return fused_computation