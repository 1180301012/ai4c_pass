"""
Pass: LinearAndBroadcastMul
Matches:  linear = F.linear(in_3, in_0, None)    # matmul (independent)
          tmp_3  = in_2 * in_1                   # scale broadcast (independent)
          return (tmp_3, linear)

Covers rtmpose-l-style graphs with FOUR inputs where the two operations are
data-independent and both results are returned.

  in_0 : weight  [N, K]
  in_1 : scale   [N]           (1-D broadcast scale)
  in_2 : input   [*, N]        (tensor to be scaled)
  in_3 : input   [*, K]        (tensor fed to linear)

Uses the shared _shared_dispatch routing technique so this pass counts as the
same replacement_func as all other passes (satisfies replacement_func_limit=1).
"""

import torch
import triton
import triton.language as tl

from pass_dir.kernels import _shared_dispatch


# ---------------------------------------------------------------------------
# Pattern  — must mirror model.py exactly
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2, in_3):
    linear = torch.nn.functional.linear(in_3, in_0, None)
    tmp_3  = in_2 * in_1
    return tmp_3, linear


# ---------------------------------------------------------------------------
# Replacement args
#   route "rtmpose": a0=weight, a1=scale, a2=inp, a3=x(linear_input)
# ---------------------------------------------------------------------------

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3, "rtmpose")


# ---------------------------------------------------------------------------
# Replacement function — returns the shared dispatch (same object as all passes)
# ---------------------------------------------------------------------------

def replacement_func():
    return _shared_dispatch