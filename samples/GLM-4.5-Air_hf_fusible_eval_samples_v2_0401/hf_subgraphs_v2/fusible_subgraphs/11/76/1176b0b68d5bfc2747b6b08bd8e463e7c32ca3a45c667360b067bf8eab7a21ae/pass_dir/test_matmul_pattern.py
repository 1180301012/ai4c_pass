import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """Pattern for matmul operations - using @ operator"""
    tmp_0 = in_1 @ in_0
    return tmp_0

@torch.fx.wrap
def simple_matmul_torch(in_0, in_1):
    # Use @ operator instead of torch.matmul
    return in_1 @ in_0

def replacement_args(in_0, in_1):
    return (in_0, in_1)

def replacement_func():
    return simple_matmul_torch