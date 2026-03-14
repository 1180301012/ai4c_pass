import torch

# Pattern matching function - matches dropout with zero rate
def pattern(tmp_1):
    # MUST mirror model.py exactly: dropout with rate 0.0
    tmp_2 = torch.nn.functional.dropout(tmp_1, 0.0, False, False)
    return tmp_2

# Argument extraction function
def replacement_args(tmp_1):
    return (tmp_1,)

# Simple identity function to replace dropout with zero rate
@torch.fx.wrap
def eliminate_dropout(x):
    """
    Since dropout rate is 0.0, this is effectively an identity operation.
    Return the input unchanged.
    """
    return x

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return eliminate_dropout