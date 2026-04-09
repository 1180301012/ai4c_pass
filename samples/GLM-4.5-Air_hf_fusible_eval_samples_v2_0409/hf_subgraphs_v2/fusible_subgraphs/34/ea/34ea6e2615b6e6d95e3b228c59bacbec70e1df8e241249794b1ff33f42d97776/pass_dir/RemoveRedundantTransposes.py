import torch

def pattern(tmp_8):
    # Match the redundant transpose pattern
    tmp_9 = tmp_8.transpose(0, 1)
    tmp_10 = tmp_8.transpose(0, 1)
    return tmp_9, tmp_10

def replacement_args(tmp_8):
    return (tmp_8,)

@torch.fx.wrap
def fuse_transposes(x):
    """Optimized function that computes transpose only once and reuses it"""
    result = x.transpose(0, 1)
    
    # We need to return two identical results, so we just reuse the first one
    return result, result

def replacement_func():
    return fuse_transposes