import torch


def pattern(x):
    """
    Pattern: dropout with p=0.0 is a no-op
    """
    tmp = torch.nn.functional.dropout(x, 0.0, False, False)
    return tmp


def replacement_args(x):
    return (x,)


def replacement_func():
    def noop_dropout(x):
        return x
    return noop_dropout