import torch

@torch.fx.wrap
def optimize_dropout_noop(x):
    """
    Optimize dropout with rate=0.0 by returning the input directly
    This avoids unnecessary computation since dropout has no effect when rate=0
    """
    return x

# Pattern matching function
def pattern(x):
    """
    Match: torch.nn.functional.dropout with rate=0.0
    """
    return torch.nn.functional.dropout(x, 0.0, False, False)

# Argument extraction function  
def replacement_args(x):
    return (x,)

# Replacement function
def replacement_func():
    return optimize_dropout_noop