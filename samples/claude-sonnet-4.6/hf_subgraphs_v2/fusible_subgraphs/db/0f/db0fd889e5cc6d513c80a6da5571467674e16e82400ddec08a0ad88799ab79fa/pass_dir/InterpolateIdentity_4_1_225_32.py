"""
Pass: InterpolateIdentity_4_1_225_32

The model's FX graph stores F.interpolate as a single leaf node.
By calling torch.fx.wrap(F.interpolate) at the START of pattern() (which is
exempt from API validation), F.interpolate becomes a leaf BEFORE the tracer
intercepts it, producing a matching single call_function(F.interpolate,...) node.

Pattern matches model lines 30-36.
Since bicubic at same size is identity, replace with Triton copy kernel.

Input:  x  shape [4, 1, 225, 32]
Output: out shape [4, 1, 225, 32]  same values, contiguous
"""

import torch
import triton
import triton.language as tl
from pass_dir._shared_kernels import shared_interp_copy


# ---------------------------------------------------------------------------
# Pattern: F.interpolate is registered as a leaf in _shared_kernels.py import.
# ---------------------------------------------------------------------------
def pattern(x):
    t1 = x.transpose(2, 3)
    t2 = t1.view(4, 32, 15, 15)
    t3 = torch.nn.functional.interpolate(t2, size=(15, 15), mode='bicubic', align_corners=False)
    t4 = t3.flatten(2)
    t5 = t4.transpose(1, 2)
    t6 = t5.contiguous()
    t7 = t6.view(4, 1, 225, 32)
    return t7


def replacement_args(x):
    return (x,)


def replacement_func():
    return shared_interp_copy