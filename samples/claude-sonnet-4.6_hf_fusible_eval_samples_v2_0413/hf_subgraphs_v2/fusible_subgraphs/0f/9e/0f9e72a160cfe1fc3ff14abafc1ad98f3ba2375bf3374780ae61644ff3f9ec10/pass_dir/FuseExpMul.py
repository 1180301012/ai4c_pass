import torch
import triton
import triton.language as tl
from pass_dir._shared_kernels import _dispatch


# ── Pattern ───────────────────────────────────────────────────────────────────

def pattern(s, x):
    e   = s.exp()
    out = e * x
    return out


# ── Argument extraction ───────────────────────────────────────────────────────

def replacement_args(s, x):
    return (s, x, "expmul")


# ── Replacement entry-point ───────────────────────────────────────────────────

def replacement_func():
    return _dispatch