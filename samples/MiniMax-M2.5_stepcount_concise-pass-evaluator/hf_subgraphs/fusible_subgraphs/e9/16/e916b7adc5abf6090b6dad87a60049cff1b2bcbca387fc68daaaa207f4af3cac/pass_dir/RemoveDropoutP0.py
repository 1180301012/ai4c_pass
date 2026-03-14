import torch


def pattern(x):
    """
    Match dropout(p=0) which is a no-op and can be removed.
    When p=0, dropout returns the input unchanged.
    """
    return torch.nn.functional.dropout(x, 0.0, False, False)


def replacement_args(x):
    return (x,)


def replacement_func():
    # Direct lambda - minimal overhead
    return lambda x: x