"""
Pass: BroadcastScaleMul
Matches:  tmp_3 = in_2 * in_1
          return tmp_3

Covers the 1-D scale broadcast multiply in rtmpose-l graphs where in_1 is a
1-D weight tensor (scale vector) broadcast-multiplied against a higher-rank in_2.

Applied AFTER FuseLinearGateMul_v1/v2 so those passes already consumed the
SmolLM3/gemma multiplication chains; this pass only fires on the remaining
element-wise muls in rtmpose-l.
"""

import torch
import triton
import triton.language as tl

from pass_dir.kernels import _shared_dispatch


# ---------------------------------------------------------------------------
# Pattern  — mirrors  tmp_3 = in_2 * in_1  in model.py exactly
# ---------------------------------------------------------------------------

def pattern(in_1, in_2):
    tmp_3 = in_2 * in_1
    return tmp_3


# ---------------------------------------------------------------------------
# Replacement args
#   route "broadcastscalemul": a0=inp(in_2), a1=scale(in_1), a2=None, a3=None
# ---------------------------------------------------------------------------

def replacement_args(in_1, in_2):
    # in_2 is the higher-rank input [*, N]; in_1 is the 1-D scale [N]
    return (in_2, in_1, None, None, "broadcastscalemul")


# ---------------------------------------------------------------------------
# Replacement function — returns the shared dispatch (same object as all passes)
# ---------------------------------------------------------------------------

def replacement_func():
    return _shared_dispatch