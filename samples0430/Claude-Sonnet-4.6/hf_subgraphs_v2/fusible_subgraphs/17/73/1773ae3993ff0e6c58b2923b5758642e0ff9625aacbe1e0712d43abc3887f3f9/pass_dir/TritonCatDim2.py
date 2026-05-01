"""
Pass: fuse torch.cat((a, b, c), dim=2) into a single Triton copy kernel.
Matches all graphs across all dtypes.
Uses shared_dispatch routing so replacement_func() is identical across all passes.
"""
import torch
from pass_dir.triton_kernels import shared_dispatch


# ---------------------------------------------------------------------------
# Pattern – mirrors model.py exactly
#   torch.cat((in_2, in_5, in_3), dim=2)
# Free tensors in first-appearance order: in_2 (a), in_5 (b), in_3 (c)
# ---------------------------------------------------------------------------
def pattern(a, b, c):
    return torch.cat((a, b, c), dim=2)


def replacement_args(a, b, c):
    return (a, b, c, "cat2")


def replacement_func():
    return shared_dispatch