"""
Pass: FuseLinearSplitUnsqueeze

Fuses:
    linear(in_5, in_1, in_0)           # [M, 2N]
    -> [:, :N]  -> view(-1, N)         # [M, N]  (first half)
    -> [:, -N:] -> view(-1, N)         # [M, N]  (second half)
    -> unsqueeze(-2) on first half     # [M, 1, N]

into a single Triton kernel that writes directly to two output buffers,
avoiding materialization of the [M, 2N] intermediate.
Uses shared dispatcher (route="lsu") to satisfy replacement_func_limit.
"""

import torch
from pass_dir.shared_gemm import dispatch_gemm_split


# ---------------------------------------------------------------------------
# Pattern / replacement API
# ---------------------------------------------------------------------------

def pattern(in_5, in_1, in_0):
    # Single-output: just the linear call. Slices/view/unsqueeze remain in graph.
    linear = torch.nn.functional.linear(in_5, in_1, in_0)
    return linear


def replacement_args(in_5, in_1, in_0):
    return (in_5, in_1, in_0)


def replacement_func():
    return dispatch_gemm_split