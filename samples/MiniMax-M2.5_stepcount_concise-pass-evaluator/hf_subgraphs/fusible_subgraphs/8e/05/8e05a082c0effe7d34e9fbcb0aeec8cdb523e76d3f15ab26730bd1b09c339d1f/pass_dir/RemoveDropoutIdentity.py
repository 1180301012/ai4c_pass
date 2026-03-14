import torch


def pattern(x):
    """
    Match dropout with p=0.0 - this is a no-op and can be removed
    """
    tmp = torch.nn.functional.dropout(x, 0.0, False, False)
    return tmp


def replacement_args(x):
    return (x,)


def replacement_func():
    # Identity function - dropout with p=0.0 does nothing
    def identity(x):
        return x
    return identity