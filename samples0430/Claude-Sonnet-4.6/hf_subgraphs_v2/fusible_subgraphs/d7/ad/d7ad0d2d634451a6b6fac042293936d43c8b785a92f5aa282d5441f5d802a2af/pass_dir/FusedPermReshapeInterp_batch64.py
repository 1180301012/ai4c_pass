"""
Pass: FusedPermReshapeInterp_batch64
Matches: permute(0,2,1) -> reshape(64, -1, 16, 16) -> bilinear_interpolate(128, 128)
Target: float32/8 and float16/8 variants (in_2 shape [64, 256, 512])
"""

import torch
from pass_dir.bilinear_kernel import fused_perm_reshape_bilinear


def pattern(x):
    """
    Match: linear_out.permute(0,2,1).reshape(64,-1,16,16) + bilinear upsample to 128x128.
    x: [64, 256, 768] linear output
    """
    p = x.permute(0, 2, 1)
    r = p.reshape(64, -1, 16, 16)
    out = torch.nn.functional.interpolate(r, size=(128, 128), mode='bilinear', align_corners=False)
    return out


def replacement_args(x):
    return (x,)


def replacement_func():
    return fused_perm_reshape_bilinear