import torch


def identity_func(x):
    """Identity function for no-op optimization."""
    return x


def pattern(in_0):
    """
    Pattern matches: in_0.long()
    For int64 tensors, this is a no-op.
    """
    return in_0.long()


def replacement_args(in_0):
    return (in_0,)


def replacement_func():
    return identity_func