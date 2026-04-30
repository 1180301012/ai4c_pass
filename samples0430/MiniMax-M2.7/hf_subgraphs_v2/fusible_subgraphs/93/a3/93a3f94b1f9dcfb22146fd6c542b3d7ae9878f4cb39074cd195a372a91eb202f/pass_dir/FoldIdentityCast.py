import torch
import triton
import triton.language as tl


def pattern(x):
    """Pattern to match: x.to(torch.float32) when x is already float32"""
    return x.to(torch.float32)


def replacement_args(x):
    return (x,)


# Module-level function
def fold_cast(x):
    return x


def replacement_func():
    return fold_cast