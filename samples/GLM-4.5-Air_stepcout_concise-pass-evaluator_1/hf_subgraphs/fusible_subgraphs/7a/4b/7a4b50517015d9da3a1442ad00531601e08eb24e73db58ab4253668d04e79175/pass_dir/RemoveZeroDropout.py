import torch

def pattern(x):
    # dropout with p=0.0 should be a no-op
    return torch.nn.functional.dropout(x, 0.0, False, False)

def replacement_args(x):
    return (x,)

def replacement_func():
    def dropout_removal(x):
        # Zero dropout rate means this is just identity function
        return x
    return dropout_removal