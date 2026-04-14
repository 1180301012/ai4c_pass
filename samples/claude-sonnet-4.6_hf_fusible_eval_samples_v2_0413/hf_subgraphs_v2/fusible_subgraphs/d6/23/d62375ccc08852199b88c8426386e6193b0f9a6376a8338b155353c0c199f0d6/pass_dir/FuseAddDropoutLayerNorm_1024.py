"""
LN-only fallback: matches layer_norm((1024,)) and replaces with Triton LN.
Uses unified shared_dispatch with 6-arg signature.
"""

import torch
from pass_dir.shared_fn import shared_dispatch


def pattern(x, weight, bias):
    normed = torch.nn.functional.layer_norm(x, (1024,), weight, bias, 1e-05)
    return normed


def replacement_args(x, weight, bias):
    # d=weight, e=bias are passed but unused by ln_h1024 route
    return (x, weight, bias, weight, bias, "ln_h1024")


def replacement_func():
    return shared_dispatch