import torch
import triton
import triton.language as tl


def pattern(x):
    return x


def replacement_args(x):
    return (x,)


def replacement_func():
    return None