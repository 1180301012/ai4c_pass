import torch
from pass_dir.shared_kernels import dispatch_wrapper


def pattern(x, y):
    """
    Pattern: unsqueeze(1) + unsqueeze(0) + add operation.
    Matches the attention bias computation.
    x = original tensor to add to [B, H, W] after unsqueeze(1).unsqueeze(0)
    y = bias tensor [B, H, W]
    """
    tmp_1 = y.unsqueeze(1)
    tmp_2 = tmp_1.unsqueeze(0)
    result = x + tmp_2
    return result


def replacement_args(x, y):
    return (x, y, "bias_add_1")


def replacement_func():
    return dispatch_wrapper