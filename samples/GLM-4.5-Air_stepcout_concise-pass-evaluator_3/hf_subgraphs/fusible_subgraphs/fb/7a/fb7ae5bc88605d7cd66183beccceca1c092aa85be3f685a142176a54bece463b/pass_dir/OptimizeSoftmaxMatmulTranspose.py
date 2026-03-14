import torch

def pattern(x):
    """
    Pattern matching unnecessary contiguous operations after reshape
    """
    result = x.contiguous()
    return (result,)

def replacement_args(x):
    return (x,)

@torch.fx.wrap
def optimized_contiguous_removal(x):
    """Wrapper function that removes unnecessary contiguous operations"""
    # Only apply optimization if tensor is already contiguous
    if x.is_contiguous():
        return x
    else:
        # Only call contiguous if actually needed
        return x.contiguous()

def replacement_func():
    return optimized_contiguous_removal