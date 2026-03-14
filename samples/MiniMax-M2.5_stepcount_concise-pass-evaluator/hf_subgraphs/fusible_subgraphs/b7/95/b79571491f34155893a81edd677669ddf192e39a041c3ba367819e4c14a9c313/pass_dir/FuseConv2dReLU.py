import torch
import triton
import triton.language as tl


def pattern(x):
    """
    Match dropout with p=0.0 which is a no-op
    This eliminates the unnecessary dropout kernel launch
    """
    # Dropout with p=0.0 just returns the input unchanged
    out = torch.nn.functional.dropout(x, 0.0, False, False)
    return out


def replacement_args(x):
    """Extract arguments for replacement"""
    return (x,)


@torch.fx.wrap
def dropout_identity(x):
    """
    Dropout with p=0.0 is an identity function - returns input unchanged
    This eliminates the unnecessary dropout kernel launch
    """
    return x


def replacement_func():
    """Return the identity function"""
    return dropout_identity