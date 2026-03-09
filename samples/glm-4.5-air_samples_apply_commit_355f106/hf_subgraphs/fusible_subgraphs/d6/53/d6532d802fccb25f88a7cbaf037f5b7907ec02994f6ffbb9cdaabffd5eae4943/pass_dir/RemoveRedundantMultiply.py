import torch

def pattern(x):
    tmp_2 = x.cos()
    tmp_3 = tmp_2 * 1.0
    tmp_6 = tmp_3.to(dtype=torch.bfloat16)
    return tmp_6

def replacement_args(x):
    return (x,)

@torch.fx.wrap
def remove_redundant_multiply(x):
    return x.to(torch.bfloat16)

def replacement_func():
    return remove_redundant_multiply