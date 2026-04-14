import torch
import triton
import triton.language as tl
from pass_dir._shared_kernel import shared_fused_div_relu_square


# ── DIAGNOSTIC: test Python-level x*x ───────────────────────────────────────
def pattern(in_0):
    return in_0 * in_0


def replacement_args(in_0):
    return (in_0,)


# ── Replacement hook (shared across ALL pass variants) ───────────────────────
def replacement_func():
    return shared_fused_div_relu_square