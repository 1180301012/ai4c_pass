import torch

# Simple matmul pattern that matches successfully
def pattern(in_0, in_1):
    tmp_0 = in_1 @ in_0
    return tmp_0

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@torch.fx.wrap
def optimized_matmul(in_0, in_1):
    return in_1 @ in_0

def replacement_func():
    return optimized_matmul