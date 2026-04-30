"""
Optimization pass for: layer_norm(16)

Matches the layer_norm with normalized_shape=(16,) in SwinV2 patch embedding.
Replaces with a fast Triton kernel. The identity dropout+view+pad+permute
remain downstream in the graph unchanged.

Uses the shared swin_ln_dispatch wrapper (integer-N dispatch) so that the
output_pass_replacement_func_limit is never exceeded across both pattern passes.
"""

import torch
from pass_dir.swin_layer_norm_kernel import swin_ln_dispatch


# ---------------------------------------------------------------------------
# Pattern: layer_norm with normalized_shape=(16,)
# Only tmp_8 is returned - it is used both inside the model return AND
# as input to the downstream identity-dropout node.
# ---------------------------------------------------------------------------

def pattern(x, weight, bias):
    tmp_8 = torch.nn.functional.layer_norm(x, (16,), weight, bias, 1e-05)
    return tmp_8


# ---------------------------------------------------------------------------
# Argument extraction: pass N=16 as an integer constant
# ---------------------------------------------------------------------------

def replacement_args(x, weight, bias):
    return (x, weight, bias, 16)


# ---------------------------------------------------------------------------
# Replacement: shared single-return wrapper
# ---------------------------------------------------------------------------

def replacement_func():
    return swin_ln_dispatch