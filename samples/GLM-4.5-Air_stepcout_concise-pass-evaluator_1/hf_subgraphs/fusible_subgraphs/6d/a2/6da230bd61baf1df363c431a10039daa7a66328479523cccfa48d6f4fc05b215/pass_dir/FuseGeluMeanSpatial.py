import torch
import triton
import triton.language as tl

def pattern(in_0):
    tmp_0 = torch.nn.functional.gelu(in_0)
    tmp_1 = tmp_0.mean((2, 3), keepdim=True)
    return (tmp_0, tmp_1)

def replacement_args(in_0):
    return (in_0)

def fused_gelu_mean(in_0):
    # For now, just use the original operations to test pattern matching
    tmp_0 = torch.nn.functional.gelu(in_0)
    tmp_1 = tmp_0.mean((2, 3), keepdim=True)
    return (tmp_0, tmp_1)

def replacement_func():
    return fused_gelu_mean