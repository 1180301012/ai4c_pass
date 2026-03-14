import torch


def pattern(a, b):
    """
    Simple add pattern
    """
    return a + b


def replacement_args(a, b):
    return (a, b)


def replacement_func():
    # Identity - not a real optimization but tests the framework
    def add_impl(a, b):
        return a + b
    return add_impl