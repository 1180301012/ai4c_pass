import torch
from pass_dir.shared_kernels import shared_dispatch


def pattern(in_2):
    """
    Match: mean(dim=-2, keepdim=True) pattern.
    """
    output = in_2.mean(dim=-2, keepdim=True)
    return output


def replacement_args(in_2):
    # Append route string as last argument for dispatch
    return (in_2, "mean_dim_minus2")


def replacement_func():
    return shared_dispatch