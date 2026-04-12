import torch

def pattern(in_2):
    tmp_0 = in_2.sigmoid()
    return tmp_0

def replacement_args(in_2):
    return (in_2,)

def replacement_func():
    def simple_sigmoid(x):
        return torch.sigmoid(x)
    return simple_sigmoid