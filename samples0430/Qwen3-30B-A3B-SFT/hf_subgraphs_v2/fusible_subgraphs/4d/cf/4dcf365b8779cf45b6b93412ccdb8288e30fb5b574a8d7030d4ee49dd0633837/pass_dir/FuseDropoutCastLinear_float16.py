"""
Pass: FuseDropoutCastLinear_float16
Matches: dropout(x, p=0.0, training=False) -> to(float16) -> linear(out, weight, bias)
         dropout is identity (training=False, p=0.0), cast is no-op (x already float16).

Covers: RECT_L float16 graph
Uses shared_dispatch with route="rect_l" for unified replacement_func.
"""

import torch
from pass_dir.shared_kernels import shared_dispatch


def pattern(bias, weight, x):
    """
    Match to(float16) cast -> linear.
    Covers RECT_L graphs where dropout(p=0.0) is eliminated (training=False, p=0.0 no-op).
    The to(float16) cast is also identity since x is already float16.
    """
    to = x.to(torch.float16)
    linear = torch.nn.functional.linear(to, weight, bias)
    return linear


def replacement_args(bias, weight, x):
    return (bias, weight, x, "rect_l")


def replacement_func():
    return shared_dispatch