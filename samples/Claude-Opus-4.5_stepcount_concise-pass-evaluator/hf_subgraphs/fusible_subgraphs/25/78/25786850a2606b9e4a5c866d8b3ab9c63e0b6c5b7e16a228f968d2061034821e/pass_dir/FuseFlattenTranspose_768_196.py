import torch
import triton
import triton.language as tl

# Pattern: dropout(x, 0.0, False, False) is a no-op
# Remove the overhead of calling dropout when it does nothing

def pattern(x):
    return torch.nn.functional.dropout(x, 0.0, False, False)


def replacement_args(x):
    return (x,)


@torch.fx.wrap
def identity(x):
    """Simply return input - no operation needed"""
    return x


def replacement_func():
    return identity