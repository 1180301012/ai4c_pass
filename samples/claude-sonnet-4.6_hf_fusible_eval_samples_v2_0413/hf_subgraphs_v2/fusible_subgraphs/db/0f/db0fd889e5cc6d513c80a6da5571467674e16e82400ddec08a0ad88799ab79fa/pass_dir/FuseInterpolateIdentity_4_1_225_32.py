"""
Pass: FuseInterpolateIdentity_4_1_225_32

Tries torch._C._nn.upsample_bicubic2d(x, [15, 15], False) — the 3-argument
C-level function form (no optional scale args). This is what
torch.nn.functional.interpolate calls internally after expanding `size`
to a list. The dynamo graph may record exactly this 3-arg call.

Uses shared routing dispatch.
"""

import torch
from pass_dir.yolos_shared import yolos_dispatch


def pattern(x):
    # Direct C-level call with 3 positional args (no scale_h/scale_w)
    # matching the form torch.nn.functional.interpolate produces internally
    out = torch._C._nn.upsample_bicubic2d(x, [15, 15], False)
    return out


def replacement_args(x):
    return (x, x, "interp_same_size")


def replacement_func():
    return yolos_dispatch