"""
Pass: FuseLayerNormTranspose_768

DESIGN
------
Pattern returns a SINGLE value (tmp_8 = layer_norm output).
The framework's _replace_pattern assertion:
    assert len(match.returning_nodes) == len(copied_returning_nodes)
requires pattern outputs == replacement outputs.  Since the replacement is
always wrapped through with_dispatch_wrapper_run (→ 1 opaque FX node), the
pattern must also produce exactly 1 output.

After replacement, the two downstream .transpose(0,1) nodes in the original
graph automatically consume the Triton-computed tmp_8 replacement.

ROUTING
-------
Both pass files import _triton_ln_dispatch from the shared module so that
replacement_func() returns the SAME Python object in both passes.
This satisfies the framework's g_replacement_func singleton assertion and
allows both passes to be loaded within the replacement_func_limit.
"""

import torch
from pass_dir._layer_norm_kernel import _triton_ln_dispatch  # shared object


# ---------------------------------------------------------------------------
# Pattern – single output (tmp_8 only)
# ---------------------------------------------------------------------------

def pattern(tmp_7, in_1, in_0):
    """layer_norm only; returns the normalised tensor (single FX output)."""
    tmp_8 = torch.nn.functional.layer_norm(tmp_7, (768,), in_1, in_0, 1e-05)
    return tmp_8


# ---------------------------------------------------------------------------
# Argument extractor – append route string for dispatch
# ---------------------------------------------------------------------------

def replacement_args(tmp_7, in_1, in_0):
    return (tmp_7, in_1, in_0, "768")


# ---------------------------------------------------------------------------
# Factory – returns THE shared dispatch object (same object as 1024 pass)
# ---------------------------------------------------------------------------

def replacement_func():
    return _triton_ln_dispatch