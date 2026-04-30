import torch
import triton
import triton.language as tl


def pattern(x):
    """Pattern to match: x * 1.0 -> should be simplified to x"""
    return x * 1.0


def replacement_args(x):
    return (x,)


# Module-level function
def fold_scale(x):
    return x


def replacement_func():
    return fold_scale