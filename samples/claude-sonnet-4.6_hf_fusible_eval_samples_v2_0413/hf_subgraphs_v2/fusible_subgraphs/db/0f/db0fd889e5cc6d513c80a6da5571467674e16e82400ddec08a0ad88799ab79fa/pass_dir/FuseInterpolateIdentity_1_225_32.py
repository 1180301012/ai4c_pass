"""
Pass: FuseInterpolateIdentity_1_225_32

Tries to match bicubic interpolate via natural FX trace-through.
When FX symbolic_trace traces torch.nn.functional.interpolate(proxy,...),
it may resolve to torch._C._nn.upsample_bicubic2d as the leaf node.

Uses shared routing dispatch.
"""

import torch
from pass_dir.yolos_shared import yolos_dispatch


def pattern(x):
    out = torch.nn.functional.interpolate(x, size=(15, 15), mode='bicubic', align_corners=False)
    return out


def replacement_args(x):
    return (x, x, "interp_same_size")


def replacement_func():
    return yolos_dispatch