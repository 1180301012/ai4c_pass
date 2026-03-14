import torch


def pattern(x):
    """Match any input and return it unchanged."""
    return x


def replacement_args(x):
    return (x,)


def replacement_func():
    """Return identity function."""
    return lambda x: x