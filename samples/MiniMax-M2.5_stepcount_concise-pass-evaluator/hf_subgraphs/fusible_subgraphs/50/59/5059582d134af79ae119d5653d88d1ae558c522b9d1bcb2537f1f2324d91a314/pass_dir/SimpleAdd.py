import torch


def pattern(x, y):
    """Match multiplication - this IS in the target graph."""
    return x * y


def replacement_args(x, y):
    return (x, y)


def replacement_func():
    """Return a simple multiply function."""
    def multiply(x, y):
        return x * y
    return multiply