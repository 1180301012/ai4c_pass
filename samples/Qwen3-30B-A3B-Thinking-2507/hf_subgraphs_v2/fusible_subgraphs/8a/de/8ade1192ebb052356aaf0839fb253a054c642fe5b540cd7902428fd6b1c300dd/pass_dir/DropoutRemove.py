import torch

def pattern(x):
    return torch.nn.functional.dropout(x, p=0.1, training=False)

def replacement_args(x):
    return (x,)

def replacement_func():
    def nop(x):
        return x
    return nop