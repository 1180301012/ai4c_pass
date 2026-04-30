"""
Pass: FuseDropoutCastLinear_bfloat16
Matches: dropout(x, p=0.0, training=False) -> to(bfloat16) -> linear(out, weight, bias)
         dropout is identity (training=False, p=0.0), cast is no-op (x already bfloat16).

Covers: RECT_L bfloat16 graph
"""

import torch
from pass_dir.shared_kernels import shared_dispatch


def pattern(bias, weight, x):
    """
    Match to(bfloat16) cast -> linear.
    Covers RECT_L bfloat16 graphs where dropout(p=0.0) is eliminated (training=False).
    The to(bfloat16) cast is also identity since x is already bfloat16.
    """
    to = x.to(torch.bfloat16)
    linear = torch.nn.functional.linear(to, weight, bias)
    return linear


def replacement_args(bias, weight, x):
    return (bias, weight, x, "rect_l")


def replacement_func():
    return shared_dispatch