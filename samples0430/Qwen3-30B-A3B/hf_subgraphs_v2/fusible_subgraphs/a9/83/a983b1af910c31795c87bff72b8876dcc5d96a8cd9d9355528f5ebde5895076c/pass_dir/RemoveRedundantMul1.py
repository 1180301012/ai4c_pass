import torch

def pattern(x):
    return x * 1.0

def replacement_args(x):
    return (x,)

def replacement_func():
    @torch.fx.wrap
    def remove_mul1(x):
        return x
    return remove_mul1