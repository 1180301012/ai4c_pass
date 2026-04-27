import torch
from pass_dir.shared_qkv import shared_dispatch  # same object across both passes


# ──────────────────────────────────────────────────────────────────────────────
# Pattern / replacement API
# Fuses: x.permute(0,2,1,3)  →  contiguous head-first layout (Q or V)
# ──────────────────────────────────────────────────────────────────────────────

def pattern(x):
    """Single-output pattern: permute(0,2,1,3) only."""
    return x.permute(0, 2, 1, 3)


def replacement_args(x):
    # Route "permute" selects the permute branch inside shared_dispatch
    return (x, "permute")


def replacement_func():
    return shared_dispatch