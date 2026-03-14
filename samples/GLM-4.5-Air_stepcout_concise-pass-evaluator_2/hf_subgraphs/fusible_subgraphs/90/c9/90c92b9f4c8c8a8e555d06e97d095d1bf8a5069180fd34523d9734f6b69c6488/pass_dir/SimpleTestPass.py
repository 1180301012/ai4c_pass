import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    # Simple test - just return one of the inputs
    return in_0


def replacement_args(in_0, in_1):
    return (in_0,)


def replacement_func():
    # Just return input as-is
    def identity(x):
        return x
    return identity