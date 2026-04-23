import torch
from pass_dir.shared_kernels import dispatch_wrapper


def pattern(x):
    """
    Pattern: sigmoid followed by multiplication by constant.
    """
    sig = torch.sigmoid(x)
    result = 16 * sig
    return result


def replacement_args(x):
    return (x, "sigmoid")


def replacement_func():
    return dispatch_wrapper