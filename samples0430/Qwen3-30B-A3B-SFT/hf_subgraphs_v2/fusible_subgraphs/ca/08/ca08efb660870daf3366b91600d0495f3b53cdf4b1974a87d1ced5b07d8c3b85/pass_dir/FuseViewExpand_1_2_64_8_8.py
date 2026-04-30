import torch
import triton
import triton.language as tl

from pass_dir.shared_dispatch import _dispatch  # noqa: F401 – shared object identity


# ── Pattern ────────────────────────────────────────────────────────────────────
def pattern(in_0):
    tmp_2 = in_0.view(1, 2, 1, 8, 8)
    tmp_3 = tmp_2.expand(1, 2, 64, 8, 8)
    return tmp_3


def replacement_args(in_0):
    return (in_0, "viewexp")


# ── Replacement hook (returns the SAME _dispatch object as FuseSumDiv) ────────
def replacement_func():
    return _dispatch