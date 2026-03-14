import torch

def pattern(tmp_2):
    """Pattern for small tensor transpose [2,1] -> [1,2]"""
    tmp_3 = tmp_2.t()
    return tmp_2, tmp_3

def replacement_args(tmp_2):
    return (tmp_2,)

@torch.fx.wrap
def optimized_small_transpose(x):
    """Optimized transpose for small tensors like [2,1] -> [1,2]"""
    # For very small tensors, just use the built-in transpose - it's already optimized
    return x, x.t()

def replacement_func():
    return optimized_small_transpose