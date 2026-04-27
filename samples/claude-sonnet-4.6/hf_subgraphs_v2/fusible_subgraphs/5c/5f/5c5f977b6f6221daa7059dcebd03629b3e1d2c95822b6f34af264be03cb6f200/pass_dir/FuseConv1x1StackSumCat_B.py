"""
Pass B: Fuse conv2d(in_3,...) → stack([.],0) → sum(0) → cat([.,in_2],1)

Structural mirror of Pass A: the conv input is in_3 and the passthrough
tensor is in_2.  We simply re-order the arguments so the same Triton
wrapper (_fused_conv1x1_cat) can be reused unchanged.
"""
import torch
import triton
import triton.language as tl
from pass_dir.FuseConv1x1StackSumCat_A import _fused_conv1x1_cat


# ── Pattern ──────────────────────────────────────────────────────────────────

def pattern(in_0, in_1, in_2, in_3):
    """
    in_0  : bias   [C_out]
    in_1  : weight [C_out, C_in, 1, 1]
    in_2  : extra  [B, C_extra, H, W]   ← passthrough  (NOT conv input)
    in_3  : input  [B, C_in,  H, W]     ← conv input
    """
    conv    = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    stacked = torch.stack([conv], dim=0)
    summed  = stacked.sum(dim=0)
    result  = torch.cat([summed, in_2], 1)
    return (result,)


def replacement_args(in_0, in_1, in_2, in_3):
    # Re-order so wrapper always receives (bias, weight, conv_input, extra)
    return (in_0, in_1, in_3, in_2)


# ── Pass entry point ──────────────────────────────────────────────────────────

def replacement_func():
    return _fused_conv1x1_cat