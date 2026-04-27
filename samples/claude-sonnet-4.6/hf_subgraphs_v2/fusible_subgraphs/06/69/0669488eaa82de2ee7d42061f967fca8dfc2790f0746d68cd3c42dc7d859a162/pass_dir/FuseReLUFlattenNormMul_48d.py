"""
Pass: FuseReLUFlattenNormMul_48d
Fuses: norm_val * 0.14433756729740643 -> clamp(min=1e-5) -> x / clamped -> mul(g)
Matches the post-norm operations for the D=48 (8x6) spatial case.
The pattern inputs are: in_0 (weight), x (flattened [B,C,D]), norm_val (L2 norm [B,C,1])
"""

import torch
from pass_dir._fused_relu_norm_impl import fused_scale_clamp_div_mul_dispatch


def pattern(in_0, x, norm_val):
    tmp_4 = norm_val * 0.14433756729740643
    tmp_5 = tmp_4.clamp(min=1e-05)
    tmp_6 = x / tmp_5
    tmp_7 = tmp_6 * in_0
    return tmp_7


def replacement_args(in_0, x, norm_val):
    return (in_0, x, norm_val, "route_48")


def replacement_func():
    return fused_scale_clamp_div_mul_dispatch