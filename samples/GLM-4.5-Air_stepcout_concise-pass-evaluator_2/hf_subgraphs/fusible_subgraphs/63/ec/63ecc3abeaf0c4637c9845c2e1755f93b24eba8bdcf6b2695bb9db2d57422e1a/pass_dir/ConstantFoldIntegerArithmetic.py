import torch
import triton
import triton.language as tl

def torch_sym_sum(lst):
    """Custom implementation of sym_sum for matching purposes"""
    result = lst[0]
    for item in lst[1:]:
        result = result + item
    return result

# Monkey patch torch with custom sym_sum
torch.sym_sum = torch_sym_sum

def pattern(in_1):
    result = torch.sym_sum([-1, in_1])
    return result

def replacement_args(in_1):
    return (in_1,)

@torch.fx.wrap
def optimized_scalar_computation(in_1):
    # Based on the computation: tmp_0 = torch.sym_sum([-1, in_1])
    # This is equivalent to -1 + in_1
    # tmp_1 = tmp_0 // 4
    # tmp_2 = torch.sym_sum([1, tmp_1])
    # We can optimize this entire sequence:
    # tmp_0 = -1 + in_1 = 4 - 1 = 3 (since in_1 is always 4)
    # tmp_1 = 3 // 4 = 0  
    # tmp_2 = 1 + 0 = 1
    # But we only need to return tmp_0, so:
    tmp_0 = -1 + in_1
    return tmp_0

def replacement_func():
    return optimized_scalar_computation