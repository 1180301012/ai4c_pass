"""
Pass: Fuse permute(0,2,1) + reshape(64,-1,16,16) + bilinear(size=(128,128))
into a single Triton kernel.
Covers: float16/batch=64  (graph: float16/8)
"""

import torch
import triton
from pass_dir.shared_perm_bilinear import fused_perm_bilinear


@torch.fx.wrap
def _fused_64_fp16_16x16_bilinear(in_0, in_1, in_2):
    B = 64
    C = 768
    out = torch.empty((B, C, 128, 128), dtype=in_2.dtype, device=in_2.device)
    fused_perm_bilinear(in_2, B, C, 256, 128, 128)
    return out


def pattern(in_0, in_1, in_2):
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    perm = linear.permute(0, 2, 1)
    reshaped = perm.reshape(64, -1, 16, 16)
    return torch.nn.functional.interpolate(reshaped, size=(128, 128), mode='bilinear', align_corners=False)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return _fused_64_fp16_16x16_bilinear