import torch

def pattern(x):
    """Pattern to match: two consecutive reshape operations that can be fused"""
    tmp_1 = x.reshape(1, 124, 2, 768)
    tmp_2 = tmp_1.reshape(1, 248, 768)
    return tmp_2

def replacement_args(x):
    return (x,)

@torch.fx.wrap
def fused_reshape(x):
    """Fused reshape operation that combines two consecutive reshape operations
    This is more efficient than individual reshape calls since it eliminates
    the intermediate tensor allocation and view operation."""
    # Directly reshape from [1, 124, 1536] to [1, 248, 768] without intermediate step
    return x.reshape(1, 248, 768)

def replacement_func():
    return fused_reshape