"""
Pass: FuseConvHardSigmoidMul_v2
Fuses POST-CONV ops: (x+3.0)/6.0 + clamp(0,1) + in_2 * attn
Conv2d is left to cuBLAS (already optimal for small 1x1 conv).
Targets: MobileNetV3 float32 graphs (mmseg).

Uses shared routing dispatch from _se_dispatch so that replacement_func()
returns the same Python object as v1, satisfying output_pass_replacement_func_limit.
"""
import torch
from pass_dir._se_dispatch import se_hardsigmoid_mul_dispatch


# -----------------------------------------------------------------------
# Pattern: match the 4 element-wise ops AFTER conv2d
#   x      = conv2d output  [B, Cout, 1, 1]
#   in_2   = feature map    [B, Cout, H, W]
# -----------------------------------------------------------------------
def pattern(x, in_2):
    tmp_3 = x    + 3.0
    tmp_4 = tmp_3 / 6.0
    tmp_5 = tmp_4.clamp_(0.0, 1.0)
    tmp_6 = in_2  * tmp_5
    return tmp_6


def replacement_args(x, in_2):
    return (x, in_2, "v2")


def replacement_func():
    return se_hardsigmoid_mul_dispatch