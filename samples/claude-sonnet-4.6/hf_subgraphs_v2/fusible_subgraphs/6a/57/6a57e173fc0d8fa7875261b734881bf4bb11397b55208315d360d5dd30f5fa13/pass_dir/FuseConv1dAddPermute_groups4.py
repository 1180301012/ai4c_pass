"""
Pass: fuse conv2d(groups=4) + iadd + permute(0,2,1,3) + contiguous
into a single Triton kernel.

torch.fx.wrap(operator.iadd) is called INSIDE pattern() (which is exempt from
API validation) so FX records operator.iadd as call_function(operator.iadd, ...)
rather than falling through to __add__.
"""

import operator
import torch
import triton
import triton.language as tl
from pass_dir.conv1d_add_permute_kernel import fused_conv1d_add_permute_kernel


# ---------------------------------------------------------------------------
# Pattern
# ---------------------------------------------------------------------------
def pattern(weight, in_1, in_2):
    # Register operator.iadd as a leaf before FX traces its call below.
    # pattern() is exempt from API validation, so torch.fx.wrap is allowed here.
    torch.fx.wrap(operator.iadd)
    conv_result = torch.conv2d(in_2, weight, None, (1, 1), (32, 0), (1, 1), 4)
    in_3 = operator.iadd(in_1, conv_result)
    tmp_3 = in_3.permute(0, 2, 1, 3)
    tmp_4 = tmp_3.contiguous()
    return tmp_4


# ---------------------------------------------------------------------------
# Replacement argument extractor
# ---------------------------------------------------------------------------
def replacement_args(weight, in_1, in_2):
    return (weight, in_1, in_2)


# ---------------------------------------------------------------------------
# Replacement kernel wrapper
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_conv1d_add_permute_g4(weight, in_1, in_2):
    B, G, H, W = in_1.shape          # G == 4
    out = torch.empty(B, H, G, W, dtype=in_1.dtype, device=in_1.device)
    grid = (B * G * H,)
    fused_conv1d_add_permute_kernel[grid](
        in_1, in_2, weight, out,
        B, G, H, W,
        KSIZE=65,
        PAD=32,
    )
    return out


def replacement_func():
    return fused_conv1d_add_permute_g4