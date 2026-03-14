import torch
import triton
import triton.language as tl

def pattern(x):
    """
    Very simple pattern - just match a cos operation
    """
    return x.cos()


def replacement_args(x):
    return (x,)


@torch.fx.wrap
def simple_cos(x):
    """
    Simple replacement
    """
    return x.cos()


def replacement_func():
    return simple_cos