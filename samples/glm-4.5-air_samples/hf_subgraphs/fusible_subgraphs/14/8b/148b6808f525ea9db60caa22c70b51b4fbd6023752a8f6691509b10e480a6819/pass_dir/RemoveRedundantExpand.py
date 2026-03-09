import torch
import triton
import triton.language as tl

def pattern(x, unsqueezed):
    # Match the pattern: x -> unsqueeze(0) -> expand(1, -1)
    # where unsqueezed is the result of x.unsqueeze(0)
    expand_result = unsqueezed.expand(1, -1)
    # Return the expand result to match what would be consumed
    return expand_result

def replacement_args(x, unsqueezed):
    return (x,)

@torch.fx.wrap
def remove_redundant_expand(x):
    # The optimization: expand(1, -1) on tensor with shape (1, N) is redundant
    # When the first dimension is already 1, expand(1, -1) is a no-op
    # We can just return the unsqueezed result directly
    return x.unsqueeze(0)

def replacement_func():
    return remove_redundant_expand