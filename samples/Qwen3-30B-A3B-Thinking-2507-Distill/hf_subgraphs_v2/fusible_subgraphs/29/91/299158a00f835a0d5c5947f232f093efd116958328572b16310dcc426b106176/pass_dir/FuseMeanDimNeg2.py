import torch
from pass_dir.shared_kernels import triton_mean


# ── Pattern ──────────────────────────────────────────────────────────────────
def pattern(in_3):
    return in_3.mean(-2)


# ── Replacement: direct call to triton_mean ──────────────────────────────────
def replacement_args(in_3):
    return (in_3,)


# ── Replacement entry-point ───────────────────────────────────────────────────
def replacement_func():
    return triton_mean