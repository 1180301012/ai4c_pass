"""
Pass: FuseSEAttnScaleBias_b3_d6

Matches the SE-attention post-conv pattern:
    (conv_out + 3.0) / 6.0  clamp_(0,1)  * in_2

conv_out = torch.conv2d(...) output  [B, C, 1, 1]
in_2                                    [B, C, H, W]

Both pass files return the SAME @torch.fx.wrap function object from
replacement_func() so the framework's replacement_func_limit is satisfied.
"""

import torch
import triton
import triton.language as tl
from pass_dir.se_fused_kernel import _fused_se_dispatch


# ---------------------------------------------------------------------------
# Pattern: only the 4 element-wise ops after conv2d (partial match)
# ---------------------------------------------------------------------------
def pattern(conv_out, in_2):
    tmp_3 = conv_out + 3.0
    tmp_4 = tmp_3 / 6.0
    tmp_5 = tmp_4.clamp_(0.0, 1.0)
    tmp_6 = in_2 * tmp_5
    return tmp_6


# ---------------------------------------------------------------------------
# Argument extractor
# ---------------------------------------------------------------------------
def replacement_args(conv_out, in_2):
    # conv_out → [B,C,1,1],  in_2 → [B,C,H,W]
    return (conv_out, in_2, 3.0, 6.0)


# ---------------------------------------------------------------------------
# Replacement: returns the SHARED dispatch wrapper (same object as b1_d2)
# ---------------------------------------------------------------------------
def replacement_func():
    return _fused_se_dispatch