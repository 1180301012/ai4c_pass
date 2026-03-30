import torch

def pattern(x):
    """Pattern to match dropout with p=0.0 - this is a no-op that had better results in float16"""
    return torch.nn.functional.dropout(x, p=0.0, training=False)

def replacement_args(x):
    return (x,)

def replacement_func():
    """Return identity function - dropout with p=0.0 is just identity"""
    def identity(x):
        return x
    return identity