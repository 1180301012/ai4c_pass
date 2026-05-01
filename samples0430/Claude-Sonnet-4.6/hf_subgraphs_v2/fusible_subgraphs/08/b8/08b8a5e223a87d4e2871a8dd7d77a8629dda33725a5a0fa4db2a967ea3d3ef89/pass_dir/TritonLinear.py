"""
Pass: TritonLinear
Matches:  linear = F.linear(in_3, in_0, None)
          return linear

Covers the standalone matmul output in rtmpose-l graphs where F.linear's
result is directly returned (not consumed by another op in the matched subgraph).

Applied AFTER FuseLinearGateMul_v1/v2 so it does NOT fire on SmolLM3/gemma
graphs (their F.linear nodes are already consumed by the fusion passes).
"""

import torch
import triton
import triton.language as tl

from pass_dir.kernels import _shared_dispatch


# ---------------------------------------------------------------------------
# Pattern
# ---------------------------------------------------------------------------

def pattern(in_0, in_3):
    linear = torch.nn.functional.linear(in_3, in_0, None)
    return linear


# ---------------------------------------------------------------------------
# Replacement args
#   route "tritonlinear": a0=weight, a1=x(linear_input), a2=None, a3=None
# ---------------------------------------------------------------------------

def replacement_args(in_0, in_3):
    return (in_0, in_3, None, None, "tritonlinear")


# ---------------------------------------------------------------------------
# Replacement function — returns the shared dispatch (same object as all passes)
# ---------------------------------------------------------------------------

def replacement_func():
    return _shared_dispatch