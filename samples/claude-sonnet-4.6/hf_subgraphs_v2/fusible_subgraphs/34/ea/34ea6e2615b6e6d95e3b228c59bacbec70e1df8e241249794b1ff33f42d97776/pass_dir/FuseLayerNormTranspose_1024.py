"""
Pass: FuseLayerNormTranspose_1024

Same design as FuseLayerNormTranspose_768 (see that file for full rationale).
Pattern returns a single value (tmp_8).  Route string "1024" selects the
correct Triton kernel in the shared dispatch function.
"""

import torch
from pass_dir._layer_norm_kernel import _triton_ln_dispatch  # shared object


# ---------------------------------------------------------------------------
# Pattern – single output (tmp_8 only)
# ---------------------------------------------------------------------------

def pattern(tmp_7, in_1, in_0):
    """layer_norm only; returns the normalised tensor (single FX output)."""
    tmp_8 = torch.nn.functional.layer_norm(tmp_7, (1024,), in_1, in_0, 1e-05)
    return tmp_8


# ---------------------------------------------------------------------------
# Argument extractor – append route string for dispatch
# ---------------------------------------------------------------------------

def replacement_args(tmp_7, in_1, in_0):
    return (tmp_7, in_1, in_0, "1024")


# ---------------------------------------------------------------------------
# Factory – returns THE shared dispatch object (same object as 768 pass)
# ---------------------------------------------------------------------------

def replacement_func():
    return _triton_ln_dispatch