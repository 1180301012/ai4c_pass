import torch

def pattern(x):
    """Pattern to match no-op dropout operations with p=0.0"""
    # Match dropout with zero probability - these are no-ops
    out = torch.nn.functional.dropout(x, 0.0, False, False)
    return out

def replacement_args(x):
    """Arguments needed for the replacement"""
    return (x,)

def replacement_func():
    """Replacement function that passes through input unchanged (no-op)"""
    def identity(x):
        return x
    return identity