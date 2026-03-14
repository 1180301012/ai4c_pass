import torch
import triton
import triton.language as tl

def pattern(in_0):
    """Match just sum + unsqueeze to test"""
    tmp_0 = in_0.sum(dim=-1)
    tmp_1 = tmp_0.unsqueeze(-1)
    return tmp_1

def replacement_args(in_0):
    return (in_0,)

@torch.fx.wrap
def simple_sum_unsqueeze(in_0):
    """Simple implementation for testing"""
    tmp_0 = in_0.sum(dim=-1)
    tmp_1 = tmp_0.unsqueeze(-1)
    return tmp_1

def replacement_func():
    return simple_sum_unsqueeze