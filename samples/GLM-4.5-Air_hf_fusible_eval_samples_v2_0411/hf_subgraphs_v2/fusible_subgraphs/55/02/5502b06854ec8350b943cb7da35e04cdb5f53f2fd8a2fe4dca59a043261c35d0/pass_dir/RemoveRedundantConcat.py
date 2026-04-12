import torch

def pattern(tmp_5):
    tmp_6 = torch.cat([tmp_5], 1)
    return tmp_6

def replacement_args(tmp_5):
    return (tmp_5,)

@torch.fx.wrap
def remove_redundant_concat(x):
    return x

def replacement_func():
    return remove_redundant_concat