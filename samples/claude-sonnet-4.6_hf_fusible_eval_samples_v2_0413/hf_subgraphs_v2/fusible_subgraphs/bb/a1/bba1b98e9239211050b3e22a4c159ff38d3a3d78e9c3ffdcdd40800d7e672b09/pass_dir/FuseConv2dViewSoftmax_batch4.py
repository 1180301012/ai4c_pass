"""
Pass: Fuse conv2d (1×1, 512→1) + view(4,1,-1) + softmax(dim=-1)
Matches graphs with batch=4 (bfloat16/2).

Uses the shared routing technique so replacement_func is identical
across all batch-variant passes, avoiding replacement_func_limit drops.
"""

import torch
from pass_dir.shared_kernels import fused_conv1x1_softmax_dispatch


# ---------------------------------------------------------------------------
# Pattern  (must mirror model.py exactly)
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    t      = conv2d.view(4, 1, -1)
    out    = t.softmax(dim=-1)
    return (out,)


# ---------------------------------------------------------------------------
# Replacement
# ---------------------------------------------------------------------------
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, "batch4")


def replacement_func():
    return fused_conv1x1_softmax_dispatch