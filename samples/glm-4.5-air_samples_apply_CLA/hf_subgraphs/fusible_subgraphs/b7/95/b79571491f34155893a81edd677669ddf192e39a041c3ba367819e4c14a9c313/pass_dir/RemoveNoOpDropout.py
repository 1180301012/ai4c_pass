import torch
import triton
import triton.language as tl

@torch.fx.wrap
def identity(x):
    """Identity function - no-op dropout is equivalent to identity"""
    return x

def pattern(x):
    # Pattern: Dropout with p=0.0 - this is effectively identity
    # Matches any tensor being passed through dropout with p=0.0
    out = torch.nn.functional.dropout(x, 0.0, False, False)
    return out

def replacement_args(x):
    return (x,)

def replacement_func():
    return identity