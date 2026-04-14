"""
Pass: FuseConv2dMaxPool_3x3_stride1

Matches: conv2d(in_1, in_0, None, (1,1), (1,1), (1,1), 1) → max_pool2d(k=3, stride=2, pad=1)
This is the resnetv2_18d stem pattern (3×3 conv with stride-1 and padding-1).

Uses the shared fused Triton kernel via route_3x3.
"""

import torch
import triton
import triton.language as tl

from pass_dir.fused_kernels import (
    fused_conv_maxpool_wrapper,
    _fused_conv7x7s2p3_maxpool3x3s2p1_kernel,
    _fused_conv3x3s1p1_maxpool3x3s2p1_kernel,
)


# ---------------------------------------------------------------------------
# Pattern – must mirror model.py exactly
# ---------------------------------------------------------------------------

def pattern(in_0, in_1):
    """
    in_0 : weight  [64, 32, 3, 3]
    in_1 : input   [N,  32, H,  W]
    """
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (1, 1), (1, 1), 1)
    tmp_2  = torch.nn.functional.max_pool2d(
        conv2d, 3, 2, 1, 1, ceil_mode=False, return_indices=False)
    return (tmp_2,)


# ---------------------------------------------------------------------------
# Argument extraction – append route tag for dispatch
# ---------------------------------------------------------------------------

def replacement_args(in_0, in_1):
    # weight first, then input, then route string
    return (in_0, in_1, "route_3x3")


# ---------------------------------------------------------------------------
# Replacement function – shared dispatch wrapper (identical across all passes)
# ---------------------------------------------------------------------------

def replacement_func():
    return fused_conv_maxpool_wrapper