import torch
from pass_dir.shared_qkv import shared_dispatch  # same object across both passes


# ──────────────────────────────────────────────────────────────────────────────
# Pattern / replacement API
# Fuses: x.permute(0,2,1,3).transpose(-2,-1)  →  K^T kernel
# ──────────────────────────────────────────────────────────────────────────────

def pattern(x):
    """
    Single-output pattern: permute(0,2,1,3) followed by transpose(-2,-1).
    Matches tmp_7 → tmp_10 → tmp_13 in the model graph.
    tmp_10 is internal; tmp_13 is the only observable output.
    """
    t = x.permute(0, 2, 1, 3)
    return t.transpose(-2, -1)


def replacement_args(x):
    # Route "kt" selects the K^T branch inside shared_dispatch
    return (x, "kt")


def replacement_func():
    return shared_dispatch