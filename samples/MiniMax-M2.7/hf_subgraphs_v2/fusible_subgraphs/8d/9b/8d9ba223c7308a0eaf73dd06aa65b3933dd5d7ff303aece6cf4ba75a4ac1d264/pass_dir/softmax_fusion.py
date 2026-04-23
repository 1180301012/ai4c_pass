import torch
from pass_dir.shared_kernels import dispatch_wrapper


def pattern(x):
    """
    Pattern: softmax with dim=-1.
    Note: Must use exact keyword argument format to match model.py
    """
    return torch.nn.functional.softmax(x, dim = -1)


def replacement_args(x):
    return (x, -1, "softmax")


def replacement_func():
    return dispatch_wrapper