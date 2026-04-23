import torch


def pattern(in_1, in_2):
    """
    Match a simple multiplication pattern.
    """
    result = in_1 * in_2
    return result


def replacement_args(in_1, in_2):
    """
    Extract arguments needed for the replacement.
    """
    return (in_1, in_2)


def replacement_func():
    """
    Return the replacement function.
    """
    def mul_replacement(a, b):
        return a * b
    return mul_replacement