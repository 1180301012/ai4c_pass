import torch

def pattern(x):
    return torch.nn.functional.dropout(x, 0.0, False, False)

def replacement_args(x):
    return (x,)

def replacement_func():
    def eliminator(x):
        return x
    
    return eliminator