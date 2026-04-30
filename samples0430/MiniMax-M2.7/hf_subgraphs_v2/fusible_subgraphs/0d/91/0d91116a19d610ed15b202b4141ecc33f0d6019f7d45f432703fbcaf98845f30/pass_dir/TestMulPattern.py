import torch


def simple_mul(x, y):
    """Module level function for multiplication."""
    return x * y


def pattern(in_4, in_5):
    """
    Simple pattern to test matching.
    """
    result = in_5 * in_4
    return result


def replacement_args(in_4, in_5):
    return (in_4, in_5)


def replacement_func():
    return simple_mul