import torch

def pattern(in_0):
    tmp_0 = torch.nn.functional.silu(in_0, inplace=True)
    return tmp_0

def replacement_args(in_0):
    return (in_0,)

def replacement_func():
    def simple_silu(x):
        return torch.nn.functional.silu(x, inplace=True)
    return simple_silu