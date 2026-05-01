"""
Optimization pass: replace torch.nn.functional.linear(x, w, None) with
a fast Triton tiled matmul.  The downstream reshape / permute / unbind /
transpose are all view-ops and remain unchanged.

Single-output pattern → 1:1 replacement → no multi-output FX issues.
Works for all head counts (H=4, H=9, H=16) and all dtypes.
"""

import torch
from pass_dir.qkv_kernels import _fast_linear


# ---------------------------------------------------------------------------
# Pattern: just the linear call  (single output = contiguous [1, M, N])
# ---------------------------------------------------------------------------
def pattern(in_0, in_1):
    linear = torch.nn.functional.linear(in_1, in_0, None)
    return linear


# ---------------------------------------------------------------------------
# Argument extraction: (weight, input)
# ---------------------------------------------------------------------------
def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ---------------------------------------------------------------------------
# Replacement: Triton fast matmul (same output shape as F.linear)
# ---------------------------------------------------------------------------
def replacement_func():
    return _fast_linear