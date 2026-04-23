import torch


def pattern(x):
    return torch.nn.functional.dropout(x, 0.5, False, False)

def replacement_args(x):
    return (x,)

@torch.fx.wrap
def no_op_dropout(x):
    return x

def replacement_func():
    return no_op_dropout