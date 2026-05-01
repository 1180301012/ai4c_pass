"""
Pass: FusedPermReshapeInterp_batch12
Matches: permute(0,2,1) -> reshape(12, -1, 16, 16) -> bilinear_interpolate(128, 128)
Target: bfloat16/4 variant (in_2 shape [12, 256, 512])
"""

import torch
from pass_dir.bilinear_kernel import fused_perm_reshape_bilinear


def pattern(x):
    """
    Match: linear_out.permute(0,2,1).reshape(12,-1,16,16) + bilinear upsample to 128x128.
    x: [12, 256, 768] linear output
    """
    p = x.permute(0, 2, 1)
    r = p.reshape(12, -1, 16, 16)
    out = torch.nn.functional.interpolate(r, size=(128, 128), mode='bilinear', align_corners=False)
    return out


def replacement_args(x):
    return (x,)


def replacement_func():
    return fused_perm_reshape_bilinear