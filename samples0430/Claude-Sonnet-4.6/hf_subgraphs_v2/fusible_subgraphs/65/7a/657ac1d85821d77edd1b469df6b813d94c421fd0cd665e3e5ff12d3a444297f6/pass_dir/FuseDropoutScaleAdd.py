"""
Pass: FuseDropoutScaleAdd

Matches: dropout(p=0) → scale_multiply → residual_add
Returns: tmp_10  (SINGLE output — avoids the returning_nodes assertion)

dropout with p=0 and training=False is an identity, so this pass fuses:
    tmp_8  = dropout(conv_out, 0.0, False, False)   # identity
    tmp_9  = tmp_8 * gamma                           # layer-scale
    tmp_10 = residual + tmp_9                        # residual add
into a single Triton kernel that writes tmp_10.
"""

import torch
from pass_dir.shared_kernel import fused_dispatch


# ---------------------------------------------------------------------------
# Pattern: 3 ops, 1 output (tmp_10)
# ---------------------------------------------------------------------------

def pattern(conv_out, gamma, residual):
    tmp_8  = torch.nn.functional.dropout(conv_out, 0.0, False, False)
    tmp_9  = tmp_8 * gamma
    tmp_10 = residual + tmp_9
    return tmp_10


def replacement_args(conv_out, gamma, residual):
    # Route "scale_add" → _run_scale_add(conv_out, gamma, residual)
    return (conv_out, gamma, residual, "scale_add")


def replacement_func():
    # MUST return the SAME fused_dispatch object as all other pass files
    return fused_dispatch