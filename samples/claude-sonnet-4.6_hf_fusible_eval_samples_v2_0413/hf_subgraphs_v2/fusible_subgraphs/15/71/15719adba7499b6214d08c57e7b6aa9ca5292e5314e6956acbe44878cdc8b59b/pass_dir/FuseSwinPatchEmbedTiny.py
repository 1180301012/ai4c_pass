"""
Fused pass for Swin Transformer tiny patch embedding post-processing.
Pattern: conv2d_output.flatten(2) -> transpose(1,2) -> layer_norm([16]) -> dropout(0.0)
Returns: tmp_9   (shape [1, 256, 16])

Uses shared dispatch_swin_patch_embed from swin_kernels.py so that the
replacement_func_limit is not triggered when both Tiny and Large passes are active.
"""

import torch
from pass_dir.swin_kernels import dispatch_swin_patch_embed


# ─── Pattern and replacement interface ────────────────────────────────────────

def pattern(conv2d, in_2, in_1):
    tmp_6 = conv2d.flatten(2)
    tmp_7 = tmp_6.transpose(1, 2)
    tmp_8 = torch.nn.functional.layer_norm(tmp_7, (16,), in_2, in_1, 1e-05)
    tmp_9 = torch.nn.functional.dropout(tmp_8, 0.0, False, False)
    return tmp_9


def replacement_args(conv2d, in_2, in_1):
    return (conv2d, in_2, in_1, "tiny")


def replacement_func():
    return dispatch_swin_patch_embed