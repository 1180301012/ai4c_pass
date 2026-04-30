import torch
from pass_dir.shared_dispatch import dispatch_wrapper


def pattern(a, b):
    result = torch.cat([a, b], dim=1)
    return result


def replacement_args(a, b):
    return (a, b, "cat_dim1")


def replacement_func():
    return dispatch_wrapper