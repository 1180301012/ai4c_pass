import torch

def pattern(in_2, tmp_3):
    tmp_4 = in_2 + tmp_3
    return tmp_4

def replacement_args(in_2, tmp_3):
    return (in_2, tmp_3)

@torch.fx.wrap
def simple_add(in_2, tmp_3):
    return in_2 + tmp_3

def replacement_func():
    return simple_add