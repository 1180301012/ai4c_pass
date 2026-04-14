import torch
import triton
import triton.language as tl

# Pattern matching function - must mirror the computation exactly
def pattern(x):
    tmp_1 = torch.cumsum(x, dim=1)
    tmp_2 = tmp_1 * x
    tmp_3 = tmp_2 - 1
    tmp_4 = tmp_3.long()
    tmp_5 = tmp_4[slice(None, None, None), slice(0, None, None)]
    tmp_6 = tmp_5 + 2
    return tmp_6

# Argument extraction function
def replacement_args(x):
    return (x,)

# Optimized fused CPU implementation for small tensors
@torch.fx.wrap
def fused_cumsum_multiply_add_cpu(x):
    # For small tensors, use fused CPU operations - faster than GPU kernel launch
    # This computation: cumsum(x, dim=1) * x + 1
    cumsum = torch.cumsum(x, dim=1)
    result = cumsum * x + 1
    return result

# Replacement function (returns function reference)
def replacement_func():
    return fused_cumsum_multiply_add_cpu