import torch

def pattern(x):
    tmp_1 = x.unsqueeze(0)
    tmp_2 = tmp_1.expand(1, -1)
    return tmp_1, tmp_2

def replacement_args(x):
    return (x,)

def fuse_unsqueeze_expand(x):
    # Fuse unsqueeze(0) + expand(1, -1) into just unsqueeze(0)
    # The expand is redundant when we're expanding to the same shape
    batch_tensor = x.unsqueeze(0)
    return x, batch_tensor

def replacement_func():
    return fuse_unsqueeze_expand