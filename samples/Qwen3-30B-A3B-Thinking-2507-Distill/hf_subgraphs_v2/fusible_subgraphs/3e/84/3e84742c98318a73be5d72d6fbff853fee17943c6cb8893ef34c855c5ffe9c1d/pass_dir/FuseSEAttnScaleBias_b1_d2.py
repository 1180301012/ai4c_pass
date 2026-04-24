"""
Pass: FuseSEAttnScaleBias_b1_d2

Matches the SE-attention post-conv pattern (4 element-wise ops only):
    (conv_out + 1.0) / 2.0  clamp_(0,1)  * in_2

conv_out = torch.conv2d(...) output  [B, C, 1, 1]
in_2                                    [B, C, H, W]

The 1x1 conv is intentionally LEFT TO cuDNN (it is already optimal).
We fuse only the 4 element-wise ops into one Triton kernel:
  - Load conv_out[bc] once  (scalar)
  - clamp((conv_out[bc] + 1.0) / 2.0, 0, 1)
  - Multiply that scalar with x2[bc, :]  (broadcast)

Both pass files return the SAME @torch.fx.wrap function object from
replacement_func() so the framework's replacement_func_limit is satisfied.
"""

import torch
import triton
import triton.language as tl
from pass_dir.se_fused_kernel import _fused_se_dispatch


# ---------------------------------------------------------------------------
# Pattern: only the 4 element-wise ops after conv2d (partial match)
# conv_out = any tensor whose output feeds the add; in_2 = the large tensor
# ---------------------------------------------------------------------------
def pattern(conv_out, in_2):
    tmp_3 = conv_out + 1.0
    tmp_4 = tmp_3 / 2.0
    tmp_5 = tmp_4.clamp_(0.0, 1.0)
    tmp_6 = in_2 * tmp_5
    return tmp_6


# ---------------------------------------------------------------------------
# Argument extractor
# ---------------------------------------------------------------------------
def replacement_args(conv_out, in_2):
    # conv_out → [B,C,1,1],  in_2 → [B,C,H,W]
    # Pass as: (scale_source, x2, scale_bias, scale_div)
    return (conv_out, in_2, 1.0, 2.0)


# ---------------------------------------------------------------------------
# Replacement: returns the SHARED dispatch wrapper (same object as b3_d6)
# ---------------------------------------------------------------------------
def replacement_func():
    return _fused_se_dispatch