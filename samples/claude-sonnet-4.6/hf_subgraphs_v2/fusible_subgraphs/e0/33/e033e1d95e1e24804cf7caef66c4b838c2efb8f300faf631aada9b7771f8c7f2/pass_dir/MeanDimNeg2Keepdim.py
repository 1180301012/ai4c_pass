import torch
from pass_dir.conv1x1_shared import dispatch_wrapper


# ── Pass interface ─────────────────────────────────────────────────────────────
def pattern(x):
    return x.mean(dim=-2, keepdim=True)


def replacement_args(x):
    # route "mean": a=x, b=None (unused), c=None (unused)
    return (x, None, None, "mean")


def replacement_func():
    return dispatch_wrapper