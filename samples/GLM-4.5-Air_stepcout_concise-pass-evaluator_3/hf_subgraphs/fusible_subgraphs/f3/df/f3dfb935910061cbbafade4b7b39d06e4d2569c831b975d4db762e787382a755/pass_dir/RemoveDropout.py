import torch


# Pattern matching function - matches dropout with p=0.0
def pattern(x):
    tmp = torch.nn.functional.dropout(x, 0.0, False, False)
    return tmp


def replacement_args(x):
    return (x,)


def dropout_replacement(x):
    # When p=0.0, dropout is a no-op, just return input
    return x


def replacement_func():
    return dropout_replacement