import torch
import triton
import triton.language as tl
from pass_dir._shared_kernels import _dispatch


# ── Pattern ───────────────────────────────────────────────────────────────────

def pattern(x):
    n   = x.norm(p=2, dim=-1, keepdim=True)
    out = x / n
    return out


# ── Argument extraction ───────────────────────────────────────────────────────
# arg1 is a duplicate of arg0 so _dispatch always receives three arguments;
# the "l2norm" branch ignores arg1.

def replacement_args(x):
    return (x, x, "l2norm")


# ── Replacement entry-point ───────────────────────────────────────────────────

def replacement_func():
    return _dispatch