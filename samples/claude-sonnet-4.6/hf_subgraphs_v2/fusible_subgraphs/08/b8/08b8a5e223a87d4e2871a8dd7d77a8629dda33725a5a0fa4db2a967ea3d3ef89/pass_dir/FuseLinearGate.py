"""
FuseLinearGate: Fuses F.linear(x, w) followed by gate * linear_result into a
single Triton kernel.  Uses the shared routing dispatcher so the framework
sees only ONE unique replacement_func across all passes.

Matches graphs like SmolLM3-3B / gemma:
    linear = F.linear(in_1, in_0, None)
    tmp_2  = in_2 * linear
    return (tmp_2,)
"""

import torch
from pass_dir._shared_kernels import shared_dispatch


# ---------------------------------------------------------------------------
# Pattern
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2):
    linear = torch.nn.functional.linear(in_1, in_0, None)
    return in_2 * linear  # plain tensor — NOT (tmp_2,)


# ---------------------------------------------------------------------------
# replacement_args – append route string; pad to 4 tensor slots with in_0
# ---------------------------------------------------------------------------

def replacement_args(in_0, in_1, in_2):
    # Exactly 3 args matching pattern inputs; a2=in_2 (gate, ndim>=3) → gate route
    return (in_0, in_1, in_2)


# ---------------------------------------------------------------------------
# replacement_func – returns the SAME shared_dispatch as FuseLinearScale
# ---------------------------------------------------------------------------

def replacement_func():
    return shared_dispatch