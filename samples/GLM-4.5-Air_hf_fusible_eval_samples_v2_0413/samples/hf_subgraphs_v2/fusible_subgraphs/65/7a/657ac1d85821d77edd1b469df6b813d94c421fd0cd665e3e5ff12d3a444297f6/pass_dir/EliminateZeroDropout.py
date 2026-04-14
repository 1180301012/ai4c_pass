import torch

def pattern(x):
    return torch.nn.functional.dropout(x, 0.0, False, False)

def replacement_args(x):
    return (x,)

def replacement_func():
    # Dropout with p=0.0 is identity, so return input directly
    return lambda x: x