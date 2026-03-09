import torch
import triton
import triton.language as tl

# Pattern matching function - matches division followed by transpose
def pattern(in_0):
    tmp_0 = in_0 / 1.6817928305074292
    tmp_1 = tmp_0.transpose(-1, -2)
    return tmp_1

# Extract arguments for replacement
def replacement_args(in_0):
    return (in_0,)

# Direct inline implementation - avoid function call overhead
# by keeping operations inlined
@torch.fx.wrap
def fused_div_transpose_wrapper(in_0):
    # Direct computation without intermediate variable
    # This is semantically identical to the original but avoids storing tmp_0
    return in_0.__truediv__(1.6817928305074292).transpose(-1, -2)


def replacement_func():
    return fused_div_transpose_wrapper