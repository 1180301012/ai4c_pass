"""
Pass: FuseLinearGateMul_v1
Matches:  linear = F.linear(in_1, in_0, None)
          tmp_2  = in_2 * linear
          return (tmp_2,)

Covers SmolLM3-style graphs where the LINEAR INPUT is in_1 and the GATE is in_2.

Uses the shared _shared_dispatch routing technique so this pass counts as the
same replacement_func as all other passes (satisfies replacement_func_limit=1).
"""

import torch
import triton
import triton.language as tl

from pass_dir.kernels import _shared_dispatch


# ---------------------------------------------------------------------------
# Pattern
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2):
    linear = torch.nn.functional.linear(in_1, in_0, None)
    tmp_2  = in_2 * linear
    return tmp_2


# ---------------------------------------------------------------------------
# Replacement args
#   route "smollm3": a0=weight, a1=gate(=in_2), a2=x(=in_1), a3=None
# ---------------------------------------------------------------------------

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_2, in_1, None, "smollm3")


# ---------------------------------------------------------------------------
# Replacement function — returns the shared dispatch (same object as all passes)
# ---------------------------------------------------------------------------

def replacement_func():
    return _shared_dispatch