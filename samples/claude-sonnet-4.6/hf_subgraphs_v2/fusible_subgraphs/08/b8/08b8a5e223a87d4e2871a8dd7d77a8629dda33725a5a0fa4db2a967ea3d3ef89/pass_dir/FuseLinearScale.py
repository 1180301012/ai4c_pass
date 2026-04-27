"""
FuseLinearScale: Handles the rtmpose-l pattern where a linear projection and
an independent elementwise scale are returned as two separate outputs.
Uses the shared routing dispatcher so the framework sees only ONE unique
replacement_func across all passes.

    linear = F.linear(in_3, in_0, None)  # in_3:[B,17,512], in_0:[256,512]
    tmp_3  = in_2 * in_1                 # in_2:[B,17,256], in_1:[256]
    return (tmp_3, linear)
"""

import torch
from pass_dir._shared_kernels import shared_dispatch


# ---------------------------------------------------------------------------
# Pattern
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2, in_3):
    linear = torch.nn.functional.linear(in_3, in_0, None)
    tmp_3 = in_2 * in_1
    return (tmp_3, linear)


# ---------------------------------------------------------------------------
# replacement_args – append route string
# ---------------------------------------------------------------------------

def replacement_args(in_0, in_1, in_2, in_3):
    # dispatcher routes by a1.ndim (== 1 for scale vector)
    return (in_0, in_1, in_2, in_3)


# ---------------------------------------------------------------------------
# replacement_func – returns the SAME shared_dispatch as FuseLinearGate
# ---------------------------------------------------------------------------

def replacement_func():
    return shared_dispatch