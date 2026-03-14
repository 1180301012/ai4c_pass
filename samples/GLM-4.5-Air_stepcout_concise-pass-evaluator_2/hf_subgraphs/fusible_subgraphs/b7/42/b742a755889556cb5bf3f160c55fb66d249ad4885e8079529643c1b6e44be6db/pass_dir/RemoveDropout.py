import torch
import triton
import triton.language as tl

def pattern(x):
    """
    Pattern matching dropout with p=0.0 which is a no-op
    This matches: torch.nn.functional.dropout(input, 0.0, False, False)
    """
    return torch.nn.functional.dropout(x, 0.0, False, False)

def replacement_args(x):
    return (x,)

# Simple replacement - dropout with p=0.0 does nothing, return input unchanged
def dropout_noop_replacement(x):
    return x

def replacement_func():
    return dropout_noop_replacement