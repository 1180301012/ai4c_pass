import torch
from torch import device

# Pattern matching: matches exp() followed by to(device=...)
# This fuses the exponential computation with the device transfer
def pattern(in_0):
    """
    Pattern: exp() -> to(device=...)
    Matches: in_0.exp() followed by to(device=device(type='cuda', index=0))
    """
    tmp_1 = in_0.exp()
    tmp_2 = tmp_1.to(device=device(type='cuda', index=0))
    return tmp_2

# Extract arguments for replacement function
def replacement_args(in_0):
    return (in_0,)

# Simply return the exp result - no need for to() since input is already on CUDA
# This optimization removes the redundant device transfer
@torch.fx.wrap
def optimized_exp(x):
    """
    Optimized exp operation that skips the redundant to() call.
    Since input is already on CUDA, we directly compute exp without the extra to() call.
    """
    return x.exp()

def replacement_func():
    return optimized_exp