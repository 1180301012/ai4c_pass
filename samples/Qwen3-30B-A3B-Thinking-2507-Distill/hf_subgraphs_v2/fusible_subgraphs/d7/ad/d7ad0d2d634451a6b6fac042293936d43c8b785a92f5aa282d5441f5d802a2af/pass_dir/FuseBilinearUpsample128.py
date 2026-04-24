"""
Pass: replace bilinear upsample [B,C,16,16] → [B,C,128,128] with a Triton kernel.
Matches the interpolate() call in all 6 target graphs.
Uses the same shared dispatch_fused_linear from fused_linear_kernel.py.
"""
import torch
from pass_dir.fused_linear_kernel import dispatch_fused_linear


def pattern(x):
    """
    Matches: torch.nn.functional.interpolate(x, size=(128,128), mode='bilinear', align_corners=False)
    """
    return torch.nn.functional.interpolate(x, size=(128, 128), mode='bilinear', align_corners=False)


def replacement_args(x):
    # Route "bilinear": dispatch_fused_linear calls the bilinear kernel
    return (x, "", "", "bilinear")


def replacement_func():
    return dispatch_fused_linear