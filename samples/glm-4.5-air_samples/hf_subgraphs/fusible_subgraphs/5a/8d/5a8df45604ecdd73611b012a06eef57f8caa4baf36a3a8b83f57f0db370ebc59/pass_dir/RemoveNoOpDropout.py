import torch

# Pattern matching function that matches dropout with p=0.0
def pattern(x):
    # Dropout with p=0.0 is effectively a no-op
    # This matches the pattern from the computation graphs
    out = torch.nn.functional.dropout(x, 0.0, False, False)
    return out

# Argument extraction function
def replacement_args(x):
    return (x,)

# Simple replacement - just return the input directly since dropout with p=0.0 is identity
def replacement_func():
    def identity_dropout(x):
        return x
    return identity_dropout