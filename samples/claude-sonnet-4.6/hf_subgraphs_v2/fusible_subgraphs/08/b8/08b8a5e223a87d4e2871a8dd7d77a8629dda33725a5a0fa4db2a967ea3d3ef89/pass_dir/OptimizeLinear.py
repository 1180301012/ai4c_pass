"""
OptimizeLinear: Replaces a standalone F.linear(in_1, in_0, None) call with a
Triton tiled-matmul kernel.  Uses the same shared_dispatch as FuseLinearGate
so the framework counts only ONE unique replacement_func.

Matches rtmpose-l graphs where the linear output is NOT immediately gated:
    linear = F.linear(in_3, in_0, None)   ← matches this
    tmp_3  = in_2 * in_1                   ← left as-is (cheap)
    return (tmp_3, linear)

Route detection: replacement_args passes in_0 (2-D weight matrix) as dummy
a2 and a3.  shared_dispatch sees a2.ndim == 2 → takes the "linear-only" route
which returns a plain tensor matching the pattern's single output.
"""

import torch
from pass_dir._shared_kernels import shared_dispatch


# ---------------------------------------------------------------------------
# Pattern  (single-output: returns plain tensor)
# ---------------------------------------------------------------------------

def pattern(in_0, in_1):
    return torch.nn.functional.linear(in_1, in_0, None)


# ---------------------------------------------------------------------------
# replacement_args
# a2 = a3 = in_0 (2-D weight matrix) → dispatcher takes linear-only route
# ---------------------------------------------------------------------------

def replacement_args(in_0, in_1):
    # 3 args: a2 = in_0 (2-D weight matrix) → linear-only route in dispatcher
    return (in_0, in_1, in_0)


# ---------------------------------------------------------------------------
# replacement_func  (identical to FuseLinearGate → same function object)
# ---------------------------------------------------------------------------

def replacement_func():
    return shared_dispatch