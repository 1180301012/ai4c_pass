import torch
import triton
import triton.language as tl


def pattern(in_2, in_3):
    """
    This pass does nothing - we use FuseCatInterpolateStack instead
    """
    pass


def replacement_args(in_2, in_3):
    return (in_2, in_3)


def replacement_func():
    pass