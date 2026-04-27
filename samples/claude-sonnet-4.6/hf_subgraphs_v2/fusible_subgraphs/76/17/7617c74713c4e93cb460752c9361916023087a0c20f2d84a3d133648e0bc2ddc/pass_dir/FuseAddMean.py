"""
Pass: FuseAddMean
Matches: element-wise add + spatial mean(dims=(2,3)) on two [B,C,H,W] tensors.
Replaces with a single Triton kernel that fuses both ops into one memory pass.
Returns a SINGLE output (the mean result), avoiding the multi-output assertion.
"""
import torch
from pass_dir.shared_ops import shared_dispatch  # same object in every pass file


# ---------------------------------------------------------------------------
# Pattern: add + mean  →  single [B,C] output
# Must mirror model.py EXACTLY (positional args, same op variants, same dataflow).
# ---------------------------------------------------------------------------

def pattern(in_4, in_5):
    tmp_4 = in_5 + in_4
    tmp_5 = tmp_4.mean((2, 3), keepdim=False)
    return tmp_5


def replacement_args(in_4, in_5):
    # Append route string as the last element so shared_dispatch can route correctly
    return (in_4, in_5, "add_mean")


def replacement_func():
    return shared_dispatch